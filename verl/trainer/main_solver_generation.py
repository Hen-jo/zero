"""
main_solver_generation.py

Run solver inference on the full challenger gen data to produce
(question, solver_answer, ground_truth, is_correct) pairs for verifier training.

Unlike main_generation.py (challenger-only), this script:
  - Preserves the original ground truth from the input dataset
  - Extracts the solver's forward-reasoning answer (<answer>...</answer>)
  - Computes EM and saves is_correct flag
  - Outputs a verifier-compatible parquet
"""

import os
import re
import string

import hydra
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from tqdm import trange
from pprint import pprint
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred_norm = normalize_answer(prediction)
    for ga in golden_answers:
        if normalize_answer(str(ga)) == pred_norm:
            return True
    return False


def extract_answer(response: str) -> str:
    matches = ANSWER_PATTERN.findall(response)
    return matches[-1].strip() if matches else ""


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_solver_generation(config)


def run_solver_generation(config):
    if not ray.is_initialized():
        temp_dir = os.path.join(os.getcwd(), "tmp/ray")
        if len(temp_dir) > 64:
            temp_dir = "/tmp/ray"
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
            _temp_dir=temp_dir,
        )
    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = pd.read_parquet(config.data.path)
    print(f"[solver_gen] Loaded {len(dataset)} samples from {config.data.path}")

    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [chat.tolist() for chat in chat_lst]

    # Preserve original ground truths
    orig_ground_truths = []
    for rm in dataset["reward_model"]:
        try:
            orig_ground_truths.append(list(rm["ground_truth"]["target"]))
        except Exception:
            orig_ground_truths.append([])

    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes
    )
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        config=config.actor_rollout_ref,
        role="rollout",
    )
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()
    wg.load_checkpoint(
        os.path.join(config.ckpt_path, "actor"), del_local_after_load=False
    )

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)

    all_solver_answers = []
    all_is_correct = []

    for batch_idx in trange(num_batch):
        start = batch_idx * config_batch_size
        end = min(start + config_batch_size, total_samples)
        batch_chat_lst = chat_lst[start:end]
        batch_gts = orig_ground_truths[start:end]

        gen_batch = defaultdict(list)
        for idx, messages in enumerate(batch_chat_lst):
            row_dict = {}
            raw_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            model_inputs = tokenizer(
                raw_prompt, return_tensors="pt", add_special_tokens=False
            )
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config.actor_rollout_ref.rollout.prompt_length,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=True,
                truncation="error",
            )
            position_ids = compute_position_id_with_mask(attention_mask)

            row_dict["index"] = f"{batch_idx}_{idx}"
            row_dict["raw_prompt"] = messages
            row_dict["full_prompts"] = raw_prompt
            row_dict["input_ids"] = input_ids[0]
            row_dict["attention_mask"] = attention_mask[0]
            row_dict["position_ids"] = position_ids[0]
            row_dict["tools_kwargs"] = {
                "search": {
                    "create_kwargs": {
                        "data_source": "search_zero",
                        "ground_truth": "",
                        "question": "",
                    }
                }
            }
            row_dict["interaction_kwargs"] = {}
            for key, value in row_dict.items():
                gen_batch[key].append(value)

        for key in gen_batch:
            if isinstance(gen_batch[key][0], torch.Tensor):
                gen_batch[key] = torch.stack(gen_batch[key])
            else:
                gen_batch[key] = np.array(gen_batch[key])

        gen_batch = DataProto.from_single_dict(gen_batch)
        # n=1 for solver inference (greedy)
        gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, wg.world_size)
        output_padded = wg.generate_sequences(gen_batch)
        outputs = unpad_dataproto(output_padded, pad_size=pad_size)

        actual_batch_size = end - start
        for i in range(actual_batch_size):
            response = tokenizer.decode(
                outputs.batch["responses"][i], skip_special_tokens=True
            )
            solver_ans = extract_answer(response)
            gt = batch_gts[i]
            is_correct = em_check(solver_ans, gt) if gt else False
            all_solver_answers.append(solver_ans)
            all_is_correct.append(is_correct)

            print(f"[{start+i}] pred={solver_ans!r:40s} | correct={is_correct}")

    # ── Save output ───────────────────────────────────────────────────────
    out_df = dataset.copy()
    out_df["solver_answer"] = all_solver_answers
    out_df["is_correct"] = all_is_correct

    acc = sum(all_is_correct) / max(len(all_is_correct), 1)
    print(f"[solver_gen] Solver EM accuracy: {acc:.2%} ({sum(all_is_correct)}/{len(all_is_correct)})")

    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    out_df.to_parquet(config.data.output_path)
    print(f"[solver_gen] Saved {len(out_df)} rows → {config.data.output_path}")


if __name__ == "__main__":
    main()

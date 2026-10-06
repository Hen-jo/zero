# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""

import os
from copy import deepcopy
import hydra
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

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

from verl.prompts import *
from verl.custom_reward.reward_function import compute_challenger_format_scores
from verl.utils.verifier_contracts import make_item_id, make_source_id, extract_source_document


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
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
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    question_word_limit = config.data.get("question_word_limit", 50)
    if isinstance(question_word_limit, bool) or not isinstance(question_word_limit, int) or question_word_limit < 0:
        raise ValueError("data.question_word_limit must be a nonnegative integer (0 preserves full questions)")

    local_path = copy_to_local(
        config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
    )
    trust_remote_code = config.data.get("trust_remote_code", False)

    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = pd.read_parquet(config.data.path)
    if dataset.empty:
        raise ValueError("Generation input is empty")
    if config.data.partition is not None:
        partition = int(config.data.partition)
        assert 0 < partition <= 5
        start = len(dataset) * (partition - 1) // 5
        end = len(dataset) * partition // 5
        dataset = dataset.iloc[start:end]
        print(f"Using partition {partition}/5, from {start} to {end}")
    
    dataset = dataset.reset_index(drop=True)
    if dataset.empty:
        raise ValueError("Selected generation partition is empty")
    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [chat.tolist() if hasattr(chat, "tolist") else chat for chat in chat_lst]
    all_hops = [int(x.split("_")[-1]) for x in dataset.data_source]
    source_documents = [extract_source_document(chat) for chat in chat_lst]

    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes
    )
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        config=config.actor_rollout_ref,
        role="rollout"
    )

    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()
    wg.load_checkpoint(
        os.path.join(config.ckpt_path, "actor"), del_local_after_load=False,
    )

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)

    all_questions, all_answers, all_contexts = [], [], []
    selected_provenance = []
    for batch_idx in trange(num_batch):
        gen_batch = defaultdict(list)
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        gen_batch["hops"] = all_hops[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
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

            row_dict["index"] = str(batch_idx)+"_"+str(idx)
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
        gen_batch = gen_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)
        gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, wg.world_size)
        
        output_padded = wg.generate_sequences(gen_batch)
        outputs = unpad_dataproto(output_padded, pad_size=pad_size)

        for i in range(0, len(outputs), config.actor_rollout_ref.rollout.n):
            cur_batch = outputs.batch[i:i+config.actor_rollout_ref.rollout.n]
            cur_hops = gen_batch.non_tensor_batch["hops"][i:i+config.actor_rollout_ref.rollout.n]

            raw_messages = [
                tokenizer.decode(cur_batch["input_ids"][i]) for i in range(len(cur_batch))
            ]
            responses = [
                tokenizer.decode(
                    cur_batch["responses"][i], skip_special_tokens=True,
                ) for i in range(len(cur_batch))
            ]

            format_scores, raw_qs, raw_ans = compute_challenger_format_scores(
                raw_messages, responses, cur_hops, return_qa=True,
            )
            sample_idx = np.argmax(format_scores)

            original_question = raw_qs[sample_idx].strip()
            question = (" ".join(original_question.split()[:question_word_limit])
                        if question_word_limit else original_question)
            if not question or not raw_ans[sample_idx].strip():
                raise ValueError("Selected generation has an empty question/answer; refusing unverifiable data")
            selected_provenance.append({
                "original_question": original_question,
                "solver_question": question,
                "question_word_limit": question_word_limit,
                "question_truncated": bool(question_word_limit and len(original_question.split()) > question_word_limit),
                "selected_candidate_index": int(sample_idx),
                "selected_format_score": float(format_scores[sample_idx]),
                "candidate_format_scores": [float(score) for score in format_scores],
                "generation_checkpoint": str(config.ckpt_path),
            })
            all_questions.append(question)
            all_answers.append(raw_ans[sample_idx])
            all_contexts.append(raw_messages[sample_idx])
            print("Question: "+ question)
            print("Answer: "+ raw_ans[sample_idx])

    if len(all_questions) != len(dataset):
        raise ValueError("Generation returned a different number of selected candidates than input rows")
    # Assemble independent records; avoid chained pandas assignment and nested aliasing.
    records = []
    seen_ids = set()
    for idx, (question, answer, context, provenance) in enumerate(
        zip(all_questions, all_answers, all_contexts, selected_provenance)
    ):
        record = deepcopy(dataset.iloc[idx].to_dict())
        source_document = source_documents[idx]
        item_id = make_item_id(question, [answer], source_document)
        if item_id in seen_ids:
            raise ValueError(f"Duplicate generated item_id {item_id}; deduplicate explicitly before training")
        seen_ids.add(item_id)
        record["prompt"] = [{"role": "user", "content": DEFAULT_SOLVER_PREFIX.format(question=question)}]
        record["reward_model"]["ground_truth"]["target"] = [answer]
        record["item_id"] = item_id
        record["source_id"] = make_source_id(source_document)
        record["question"] = question
        record["reference_answers"] = [answer]
        metadata = record.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        metadata.update(provenance)
        metadata.update({"raw_context": context, "source_document": source_document,
                         "proposed_answer": answer, "schema_version": "verifier_pipeline_v1"})
        record["metadata"] = metadata
        records.append(record)
    dataset = pd.DataFrame(records)

    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir or ".", exist_ok=True)
    dataset.to_parquet(config.data.output_path, index=False)


if __name__ == "__main__":
    main()

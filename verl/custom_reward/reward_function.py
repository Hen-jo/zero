import re
import json
import string
import logging

import os
from collections import defaultdict, Counter

import asyncio
from copy import deepcopy

# MultiTurnRewardRollout은 임포트 시점에 sglang 전체를 로드하고
# triton이 CUDA 드라이버를 초기화하려 함. Ray CPU worker에는 CUDA가 없으므로
# 모듈 레벨 임포트를 피하고 함수 내부에서 지연 임포트함.
from verl.prompts import *


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SOLVER_PROMPT_MAX_LENGTH = 512


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def count_valid_tool_calls(response):
    valid_count = 0
    tool_matches = re.findall(TOOL_CALL_PATTERN, response, re.DOTALL)
    for match in tool_matches:
        try:
            parsed = json.loads(match.strip())
            if (
                isinstance(parsed, dict)
                and parsed.get("name") == "search"
                and isinstance(parsed.get("arguments"), dict)
                and isinstance(parsed["arguments"].get("query_list"), list)
                and all(isinstance(q, str) for q in parsed["arguments"]["query_list"])
            ):
                valid_count += 1
        except:
            continue

    return valid_count


def compute_difficulty_score(binary_list):
    assert len(binary_list) > 1
    n = len(binary_list) 
    num_ones = sum(binary_list)
    if num_ones == 0 or num_ones == n:
        return 0.0
    score = (n - num_ones) / (n - 1)    
    return score


def compute_challenger_format_scores(raw_messages, responses, hops, return_qa=False):
    think_rewards = []
    assistant_turns = [re.findall(ASSISTANT_PATTERN, m, re.DOTALL) for m in raw_messages]
    for messages in assistant_turns:
        reward = 0.0
        for message in messages:
            if re.match(THINK_PATTERN, message.strip(), re.DOTALL):
                reward += 1.0

        think_rewards.append(reward / max(1, len(messages)))
    
    raw_docs, tool_rewards = [], []
    for messages, response, hop in zip(raw_messages, responses, hops):
        tool_calls = count_valid_tool_calls(response)
        doc = re.findall(SOURCE_PATTERN, messages, re.DOTALL)[0]
        doc_matches = re.findall(TOOL_RESPONSE_PATTERN, messages, re.DOTALL)
        
        reward = 1.0 if hop == 1 else 0.0
        if tool_calls == len(doc_matches) and tool_calls > 0:
            reward = min((1 + tool_calls) / hop, 1.0)
            try:
                doc += "\n\n" + "\n\n".join([json.loads(x)["result"] for x in doc_matches])
            except:
                logger.error(f"Error parsing tool response: {doc_matches}")

        raw_docs.append(doc)
        tool_rewards.append(reward)

    raw_ans, ans_rewards = [], []
    for doc, text in zip(raw_docs, responses):
        ans_matches = re.findall(ANSWER_PATTERN, text, re.DOTALL)
        ans = ans_matches[-1].strip() if ans_matches else ""
        raw_ans.append(ans)
        
        reward = 0.0
        norm_ans = normalize_answer(ans)
        if norm_ans in ["yes", "no"] or norm_ans in normalize_answer(doc):
            if len(norm_ans) > 0 and len(norm_ans.split()) <= 5:
                reward = 1.0
            elif len(norm_ans) > 0 and len(norm_ans.split()) <= 10:
                reward = 0.5

        ans_rewards.append(reward)
    
    raw_qs = []
    for text in responses:
        q_matches = re.findall(QUESTION_PATTERN, text, re.DOTALL)
        raw_qs.append(q_matches[-1].strip() if q_matches else "")

    integrity_rewards = [len(q) > 0 and len(a) > 0 and normalize_answer(a) not in normalize_answer(q) for q, a in zip(raw_qs, raw_ans)]
    final_scores = [sum([1, f, t, a]) / 4 if i else 0 for i, f, t, a in zip(integrity_rewards, think_rewards, tool_rewards, ans_rewards)]
    # print(
    #     f"Think rewards: Avg {np.mean(think_rewards)}, Max {np.max(think_rewards)}\n"
    #     f"Tool rewards: Avg {np.mean(tool_rewards)}, Max {np.max(tool_rewards)}\n"
    #     f"Answer rewards: Avg {np.mean(ans_rewards)}, Max {np.max(ans_rewards)}\n"
    #     f"Question rewards: Avg {np.mean(integrity_rewards)}, Max {np.max(integrity_rewards)}\n"
    #     f"Final format rewards: Avg {np.mean(final_scores)}, Max {np.max(final_scores)}\n"
    # )

    if return_qa:
        return final_scores, raw_qs, raw_ans
    else:
        return final_scores


def compute_challenger_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    """Original difficulty reward, with an optional experimental validity bonus."""
    from verl.custom_reward.verifier_feedback import VerifierFeedbackConfig, finish_challenger_scores
    feedback_config = VerifierFeedbackConfig.from_kwargs(kwargs)
    # Configuration errors surface before importing/starting the rollout stack.
    import torch
    import numpy as np
    from verl import DataProto
    from verl.utils.model import compute_position_id_with_mask
    import verl.utils.torch_functional as verl_F

    batch = kwargs["data"]
    processing_class = kwargs["processing_class"]
    assert "qwen" in type(processing_class).__name__.lower()

    rollout_config = deepcopy(kwargs["config"])
    rollout_config["prompt_length"] = SOLVER_PROMPT_MAX_LENGTH

    hops = [int(x.split('_')[-1]) for x in batch.non_tensor_batch["data_source"]]
    raw_messages = [
        processing_class.decode(batch.batch["input_ids"][i]) for i in range(len(batch))
    ]
    responses = [
        processing_class.decode(
            batch.batch["responses"][i], skip_special_tokens=True,
        ) for i in range(len(batch))
    ]

    format_scores, raw_qs, raw_ans = compute_challenger_format_scores(
        raw_messages, responses, hops, return_qa=True,
    )

    gen_batch_ids, gen_batch = [], defaultdict(list)
    for idx, (s, q, a) in enumerate(zip(format_scores, raw_qs, raw_ans)):
        if s > 0:
            gen_batch_ids.append(idx)
            row_dict, messages = {}, [
                {"role": "user", "content": DEFAULT_SOLVER_PREFIX.format(question=q.strip())}
            ]

            raw_prompt = processing_class.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            model_inputs = processing_class(
                raw_prompt, return_tensors="pt", add_special_tokens=False
            )
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=rollout_config["prompt_length"],
                pad_token_id=processing_class.pad_token_id,
                left_pad=True,
                truncation="left",  # in case of very long questions
            )
            position_ids = compute_position_id_with_mask(attention_mask)

            row_dict["index"] = idx
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

    if len(gen_batch_ids) == 0:
        return finish_challenger_scores(
            format_scores, [], raw_qs, raw_ans, feedback_config, **kwargs,
        )

    for key in gen_batch:
        if isinstance(gen_batch[key][0], torch.Tensor):
            gen_batch[key] = torch.stack(gen_batch[key])
        else:
            gen_batch[key] = np.array(gen_batch[key])

    group_size = kwargs["reward_rollout_n"]
    gen_batch = DataProto.from_single_dict(gen_batch)
    gen_batch = gen_batch.repeat(repeat_times=group_size, interleave=True)

    async def _compute_with_rollout():
        from verl.custom_reward.reward_rollout import MultiTurnRewardRollout  # lazy: avoid triton CUDA init in CPU workers
        async with MultiTurnRewardRollout(
            config=rollout_config,
            processing_class=processing_class,
            model_name=kwargs["model_name"],
            base_url=kwargs["base_url"],
        ) as reward_rollout:
            gen_batch_output = await reward_rollout.generate_sequences(gen_batch)
            return gen_batch_output

    loop = asyncio.get_event_loop()
    outputs = loop.run_until_complete(_compute_with_rollout())

    extracted_preds = []
    last_turn_responses = [
        x["messages"][-1].content for x in outputs.non_tensor_batch["messages"]
    ]
    for raw_response in last_turn_responses:
        extracted = re.findall(ANSWER_PATTERN, raw_response, re.DOTALL)
        if len(extracted) > 0:
            extracted_preds.append(extracted[-1].strip())
        else:
            extracted_preds.append(raw_response.strip())

    solver_scores = []
    grouped_preds = [
        extracted_preds[i:i + group_size] for i in range(0, len(extracted_preds), group_size)
    ]
    ground_truths = [x for idx, x in enumerate(raw_ans) if idx in gen_batch_ids]
    for preds, gt in zip(grouped_preds, ground_truths):
        em_scores = [em_check(pred, gt) for pred in preds]
        solver_scores.append(compute_difficulty_score(em_scores))

    final_scores = [0.5 * x for x in format_scores]
    for idx, score in zip(gen_batch_ids, solver_scores):
        final_scores[idx] += score

    print(
        f"🚀 Raw format rewards: Avg {np.mean(format_scores)}, Max {np.max(format_scores)}\n"
        f"🚀 Final rewards: Avg {np.mean(final_scores)}, Max {np.max(final_scores)}\n"
        f"🧑‍🏫 Challenger question: {raw_qs[gen_batch_ids[0]]}\n"
        f"🧑‍🎓 Challenger answer: {raw_ans[gen_batch_ids[0]]}\n"
        f"🙋‍♂️ Solver responses: {grouped_preds[0]}"
    )
    return finish_challenger_scores(
        final_scores, gen_batch_ids, raw_qs, raw_ans, feedback_config, **kwargs,
    )


def compute_challenger_score_batch_growth(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    """Growth-aware challenger reward (selected by iter2; iter3 uses difficulty).

    r = 0.5 * format + difficulty(acc_curr) + lambda * max(0, acc_curr - acc_prev) * (1 - acc_curr)

    growth = max(0, acc_curr - acc_prev) * (1 - acc_curr)
      - Rewards questions where current solver improved over previous (acc_curr > acc_prev)
      - Self-regulates: (1 - acc_curr) penalizes questions that are already too easy
      - Peak reward at acc_prev=0, acc_curr=0.5

    Requires two sglang servers:
      base_url      : current solver  (iter t)   on port 8001
      prev_base_url : previous solver (iter t-1) on port 8002

    Optional verifier feedback adds an answer-validity proxy bonus without
    replacing either the difficulty or growth term. It is disabled by default.
    """
    from verl.custom_reward.verifier_feedback import VerifierFeedbackConfig, finish_challenger_scores
    feedback_config = VerifierFeedbackConfig.from_kwargs(kwargs)
    import torch
    import numpy as np
    from verl import DataProto
    from verl.utils.model import compute_position_id_with_mask
    import verl.utils.torch_functional as verl_F

    batch = kwargs["data"]
    processing_class = kwargs["processing_class"]
    assert "qwen" in type(processing_class).__name__.lower()

    rollout_config = deepcopy(kwargs["config"])
    rollout_config["prompt_length"] = SOLVER_PROMPT_MAX_LENGTH

    hops = [int(x.split('_')[-1]) for x in batch.non_tensor_batch["data_source"]]
    raw_messages = [
        processing_class.decode(batch.batch["input_ids"][i]) for i in range(len(batch))
    ]
    responses = [
        processing_class.decode(
            batch.batch["responses"][i], skip_special_tokens=True,
        ) for i in range(len(batch))
    ]

    format_scores, raw_qs, raw_ans = compute_challenger_format_scores(
        raw_messages, responses, hops, return_qa=True,
    )

    gen_batch_ids, gen_batch = [], defaultdict(list)
    for idx, (s, q, a) in enumerate(zip(format_scores, raw_qs, raw_ans)):
        if s > 0:
            gen_batch_ids.append(idx)
            messages = [{"role": "user", "content": DEFAULT_SOLVER_PREFIX.format(question=q.strip())}]
            raw_prompt = processing_class.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            model_inputs = processing_class(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=rollout_config["prompt_length"],
                pad_token_id=processing_class.pad_token_id,
                left_pad=True,
                truncation="left",
            )
            position_ids = compute_position_id_with_mask(attention_mask)
            row_dict = {
                "index": idx,
                "raw_prompt": messages,
                "full_prompts": raw_prompt,
                "input_ids": input_ids[0],
                "attention_mask": attention_mask[0],
                "position_ids": position_ids[0],
                "tools_kwargs": {"search": {"create_kwargs": {"data_source": "search_zero", "ground_truth": "", "question": ""}}},
                "interaction_kwargs": {},
            }
            for key, value in row_dict.items():
                gen_batch[key].append(value)

    if len(gen_batch_ids) == 0:
        return finish_challenger_scores(
            format_scores, [], raw_qs, raw_ans, feedback_config, **kwargs,
        )

    for key in gen_batch:
        if isinstance(gen_batch[key][0], torch.Tensor):
            gen_batch[key] = torch.stack(gen_batch[key])
        else:
            gen_batch[key] = np.array(gen_batch[key])

    group_size = kwargs["reward_rollout_n"]
    gen_batch = DataProto.from_single_dict(gen_batch)
    gen_batch = gen_batch.repeat(repeat_times=group_size, interleave=True)

    growth_lambda = kwargs.get("growth_lambda", 0.5)
    prev_base_url = kwargs["prev_base_url"]
    prev_model_name = kwargs["prev_model_name"]

    async def _compute_with_two_rollouts():
        from verl.custom_reward.reward_rollout import MultiTurnRewardRollout
        async with MultiTurnRewardRollout(
            config=rollout_config,
            processing_class=processing_class,
            model_name=kwargs["model_name"],
            base_url=kwargs["base_url"],
        ) as reward_rollout:
            curr_output = await reward_rollout.generate_sequences(gen_batch)

        async with MultiTurnRewardRollout(
            config=rollout_config,
            processing_class=processing_class,
            model_name=prev_model_name,
            base_url=prev_base_url,
        ) as reward_rollout:
            prev_output = await reward_rollout.generate_sequences(gen_batch)

        return curr_output, prev_output

    loop = asyncio.get_event_loop()
    curr_outputs, prev_outputs = loop.run_until_complete(_compute_with_two_rollouts())

    def _extract_preds(outputs):
        preds = []
        for x in outputs.non_tensor_batch["messages"]:
            raw = x["messages"][-1].content
            extracted = re.findall(ANSWER_PATTERN, raw, re.DOTALL)
            preds.append(extracted[-1].strip() if extracted else raw.strip())
        return preds

    curr_preds = _extract_preds(curr_outputs)
    prev_preds = _extract_preds(prev_outputs)

    ground_truths_filtered = [x for idx, x in enumerate(raw_ans) if idx in gen_batch_ids]
    grouped_curr = [curr_preds[i:i + group_size] for i in range(0, len(curr_preds), group_size)]
    grouped_prev = [prev_preds[i:i + group_size] for i in range(0, len(prev_preds), group_size)]

    difficulty_scores = []
    growth_scores = []
    for curr_group, prev_group, gt in zip(grouped_curr, grouped_prev, ground_truths_filtered):
        curr_em = [em_check(p, gt) for p in curr_group]
        prev_em = [em_check(p, gt) for p in prev_group]
        acc_curr = np.mean(curr_em)
        acc_prev = np.mean(prev_em)
        difficulty_scores.append(compute_difficulty_score(curr_em))
        growth_scores.append(max(0.0, acc_curr - acc_prev) * (1.0 - acc_curr))

    final_scores = [0.5 * x for x in format_scores]
    for idx, diff_score, grow_score in zip(gen_batch_ids, difficulty_scores, growth_scores):
        final_scores[idx] += diff_score + growth_lambda * grow_score

    print(
        f"🚀 Format rewards: Avg {np.mean(format_scores):.3f}\n"
        f"📊 Difficulty scores: Avg {np.mean(difficulty_scores):.3f}\n"
        f"🌱 Growth scores: Avg {np.mean(growth_scores):.3f}, Max {np.max(growth_scores):.3f}\n"
        f"🚀 Final rewards: Avg {np.mean(final_scores):.3f}\n"
        f"🧑‍🏫 Challenger Q: {raw_qs[gen_batch_ids[0]]}\n"
        f"🧑‍🎓 Challenger A: {raw_ans[gen_batch_ids[0]]}\n"
        f"🙋 Curr solver: {grouped_curr[0]}\n"
        f"🙋 Prev solver: {grouped_prev[0]}"
    )
    return finish_challenger_scores(
        final_scores, gen_batch_ids, raw_qs, raw_ans, feedback_config, **kwargs,
    )

"""
Generate Verifier training data from Challenger gen_data + Solver predictions.

Pipeline:
  iter1_gen_data.sh          → gen_data parquet  (Challenger question + answer)
  iter1_verifier_gen_data.sh → solver_pred parquet (Solver predictions)
                                        ↓
                              verifier_data.py  (this script)
                                        ↓
                               verifier/train.parquet
                          (question + solver_pred + is_correct)

Positive: solver_pred가 맞은 경우 (em_check=1) → is_correct=True
Negative: solver_pred가 틀린 경우 (em_check=0) → is_correct=False
→ 인위적 negative 불필요, Solver 실패가 자연스러운 negative
"""

import re
import os
import string
import argparse
import pandas as pd

from verl.prompts import DEFAULT_BACKWARD_VERIFIER_PREFIX, QUESTION_PATTERN, ANSWER_PATTERN


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
    return any(normalize_answer(gt) == pred_norm for gt in golden_answers)


def extract_last_tag(text, pattern):
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1].strip() if matches else ""


def make_verifier_row(question, candidate_answer, is_correct, idx):
    return {
        "data_source": "verifier",
        "prompt": [{"role": "user", "content": DEFAULT_BACKWARD_VERIFIER_PREFIX.format(
            question=question,
            candidate_answer=candidate_answer,
        )}],
        "ability": "backward-verification",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"is_correct": is_correct},
        },
        "extra_info": {"split": "train", "index": idx},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_data_path", required=True,
                        help="iter_gen_data.sh 출력 parquet (Challenger question+answer)")
    parser.add_argument("--solver_pred_path", required=True,
                        help="iter_verifier_gen_data.sh 출력 parquet (Solver 예측)")
    parser.add_argument("--local_dir", default="./data/verifier")
    # gen_data 컬럼명 (main_generation 출력 포맷에 맞게 조정)
    parser.add_argument("--gen_response_col", default="responses",
                        help="Challenger 응답 텍스트 컬럼 (<question>/<answer> 파싱용)")
    parser.add_argument("--gen_answer_col", default=None,
                        help="이미 추출된 경우: proposer answer 컬럼명")
    parser.add_argument("--gen_question_col", default=None,
                        help="이미 추출된 경우: question 컬럼명")
    parser.add_argument("--solver_response_col", default="responses",
                        help="Solver 응답 텍스트 컬럼 (<answer> 파싱용)")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    gen_df = pd.read_parquet(args.gen_data_path)
    solver_df = pd.read_parquet(args.solver_pred_path)
    print(f"gen_data rows: {len(gen_df)}, solver_pred rows: {len(solver_df)}")
    assert len(gen_df) == len(solver_df), \
        "gen_data와 solver_pred 행 수가 다릅니다. 같은 입력 데이터로 생성했는지 확인하세요."

    # Challenger 답 (ground truth) 추출
    if args.gen_question_col and args.gen_answer_col:
        proposer_qs = gen_df[args.gen_question_col].tolist()
        proposer_ans = gen_df[args.gen_answer_col].tolist()
    else:
        proposer_qs, proposer_ans = [], []
        for resp in gen_df[args.gen_response_col]:
            q = extract_last_tag(str(resp), QUESTION_PATTERN)
            a = extract_last_tag(str(resp), ANSWER_PATTERN)
            proposer_qs.append(q)
            proposer_ans.append(a)

    # Solver 예측 추출
    solver_preds = []
    for resp in solver_df[args.solver_response_col]:
        pred = extract_last_tag(str(resp), ANSWER_PATTERN)
        solver_preds.append(pred)

    # Verifier 학습 데이터 생성
    rows = []
    n_pos, n_neg, n_skip = 0, 0, 0
    for idx, (q, proposer_a, solver_pred) in enumerate(
        zip(proposer_qs, proposer_ans, solver_preds)
    ):
        if not q or not proposer_a or not solver_pred:
            n_skip += 1
            continue

        is_correct = em_check(solver_pred, proposer_a)
        rows.append(make_verifier_row(q, solver_pred, is_correct, idx))
        if is_correct:
            n_pos += 1
        else:
            n_neg += 1

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(args.local_dir, "train.parquet")
    out_df.to_parquet(out_path, index=False)

    print(f"Saved {len(out_df)} Verifier training rows → {out_path}")
    print(f"  Positive (is_correct=True) : {n_pos}")
    print(f"  Negative (is_correct=False): {n_neg}")
    print(f"  Skipped (empty Q/A)        : {n_skip}")

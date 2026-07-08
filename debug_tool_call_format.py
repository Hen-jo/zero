"""
debug_tool_call_format.py
챌린저/솔버 모델이 tool_call을 어떤 포맷으로 생성하는지 확인하는 진단 스크립트.

Usage:
  python debug_tool_call_format.py --model_path /input/output/zero_no2/checkpoints/dr-zero/challenger_iter2_ratio4321_grpo_batch_group1-5_qwen/challenger_iter2_hf
  python debug_tool_call_format.py --model_path /input/model/Qwen  # base model
"""

import argparse
import json
import re
import sys

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Searches the web for relevant information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search queries."
                }
            },
            "required": ["query_list"]
        }
    }
}

SAMPLE_DOC = (
    "(Title: Auburn University)\n"
    "Auburn University is a public research university in Auburn, Alabama. "
    "It was founded in 1856 and is the state's oldest land-grant institution. "
    "The university's president is Christopher B. Roberts."
)

CHALLENGER_PROMPT = (
    "You are an expert in question generation. Craft one challenging, deterministic question "
    "and its single, unambiguous answer based on the provided source document. "
    "The logical path must start from the document and require exactly n hops (i.e., n-1 searches) "
    "to reach the final answer.\n\n"
    "For each hop transition, issue a search query using <tool_call> ... </tool_call>.\n"
    "Search results will be provided between <tool_response> and </tool_response>.\n\n"
    "Now, generate a question and its answer with n = 2 hops starting from the following "
    f"source document: {SAMPLE_DOC}"
)

SOLVER_PROMPT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>. "
    "Question: Who is the president of the university that Christopher B. Roberts leads?"
)


def analyze_output(text: str, label: str):
    print(f"\n{'='*60}")
    print(f"[{label}] RAW OUTPUT:")
    print("-" * 60)
    print(text)
    print("-" * 60)

    tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    print(f"\n[{label}] ANALYSIS:")
    print(f"  <tool_call> 태그 개수: {len(tool_calls)}")
    for i, tc in enumerate(tool_calls):
        tc_stripped = tc.strip()
        try:
            parsed = json.loads(tc_stripped)
            print(f"  tool_call[{i}]: ✅ JSON 포맷 — {parsed}")
        except json.JSONDecodeError:
            print(f"  tool_call[{i}]: ❌ plain text — '{tc_stripped[:80]}'")

    has_question = "<question>" in text
    has_answer = "<answer>" in text
    print(f"  <question> 존재: {has_question}")
    print(f"  <answer> 존재: {has_answer}")


def run_hf_inference(model_path: str):
    """transformers로 단순 inference (tool schema 없음 — 모델 raw 동작 확인)"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("transformers / torch not available")
        sys.exit(1)

    print(f"\n모델 로딩: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    for role, prompt, label in [
        ("user", CHALLENGER_PROMPT, "CHALLENGER (no tool schema)"),
        ("user", SOLVER_PROMPT, "SOLVER (no tool schema)"),
    ]:
        messages = [{"role": role, "content": prompt}]
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False,
        )
        analyze_output(generated, label)


def run_hf_with_tools(model_path: str):
    """tool schema를 chat template에 넣어서 inference (sglang와 유사한 환경)"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("transformers / torch not available")
        sys.exit(1)

    print(f"\n모델 로딩 (with tools): {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    for prompt, label in [
        (CHALLENGER_PROMPT, "CHALLENGER (with tool schema)"),
        (SOLVER_PROMPT, "SOLVER (with tool schema)"),
    ]:
        messages = [{"role": "user", "content": prompt}]
        try:
            text_input = tokenizer.apply_chat_template(
                messages,
                tools=[TOOL_SCHEMA],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            print(f"apply_chat_template with tools failed: {e}")
            text_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False,
        )
        analyze_output(generated, label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="HF-format checkpoint path")
    parser.add_argument("--with_tools", action="store_true", default=True,
                        help="Also test with tool schema in chat template (default: True)")
    args = parser.parse_args()

    print("=" * 60)
    print("챌린저/솔버 tool call 포맷 진단 스크립트")
    print("=" * 60)
    print(f"모델: {args.model_path}")

    # 1. tool schema 없이 raw generation
    run_hf_inference(args.model_path)

    # 2. tool schema 포함
    if args.with_tools:
        run_hf_with_tools(args.model_path)


if __name__ == "__main__":
    main()

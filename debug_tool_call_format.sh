#!/bin/bash
# debug_tool_call_format.sh
# 챌린저/솔버 모델이 tool_call을 어떤 포맷으로 생성하는지 확인
#
# Usage:
#   bash debug_tool_call_format.sh /path/to/hf_checkpoint

MODEL_PATH=${1:-"/input/model/Qwen"}
LOG_FILE="debug_tool_call_$(date '+%Y%m%d_%H%M%S').log"

echo "========================================"
echo "tool call 포맷 진단"
echo "모델: $MODEL_PATH"
echo "로그: $LOG_FILE"
echo "========================================"

export MODEL_PATH

python3 - << 'PYEOF' 2>&1 | tee "$LOG_FILE"
import sys, json, re, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ["MODEL_PATH"]

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
    "Now generate a question with n = 2 hops from this source document: " + SAMPLE_DOC
)

SOLVER_PROMPT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return results between <tool_response> and </tool_response>. "
    "Question: Who is the president of the university that Christopher B. Roberts leads?"
)

def analyze(text, label):
    print(f"\n{'='*60}")
    print(f"[{label}] RAW OUTPUT:")
    print("-" * 60)
    print(text)
    print("-" * 60)
    tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    print(f"[{label}] 분석:")
    print(f"  <tool_call> 태그 수: {len(tool_calls)}")
    for i, tc in enumerate(tool_calls):
        try:
            parsed = json.loads(tc.strip())
            print(f"  tool_call[{i}]: OK JSON -- {parsed}")
        except json.JSONDecodeError:
            print(f"  tool_call[{i}]: NG plain text -- '{tc.strip()[:100]}'")
    print(f"  <question> 존재: {'<question>' in text}")
    print(f"  <answer>   존재: {'<answer>'   in text}")

print(f"\n[로딩] {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)
model.eval()

cases = [
    (CHALLENGER_PROMPT, "CHALLENGER  no-tools", False),
    (CHALLENGER_PROMPT, "CHALLENGER with-tools", True),
    (SOLVER_PROMPT,     "SOLVER      no-tools", False),
    (SOLVER_PROMPT,     "SOLVER    with-tools", True),
]

for prompt, label, use_tools in cases:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"tools": [TOOL_SCHEMA]} if use_tools else {}
    try:
        text_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
    except Exception as e:
        print(f"[WARN] tools 옵션 실패 ({e}), tools 없이 재시도")
        text_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    generated = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )
    analyze(generated, label)

PYEOF

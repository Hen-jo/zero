import os
import subprocess, sys
_sglang_root = subprocess.check_output(
    [sys.executable, "-c", "import sglang; import os; print(os.path.dirname(sglang.__file__))"],
    text=True
).strip()
SGLANG_UTILS = os.path.join(_sglang_root, "srt/utils.py")
SGLANG_QUANT = os.path.join(_sglang_root, "srt/layers/quantization/utils.py")

def patch_file(path, old, new, label):
    if not os.path.exists(path):
        print(f"[SKIP] {label}: file not found")
        return
    with open(path) as f:
        content = f.read()
    if old not in content:
        print(f"[SKIP] {label}: pattern not found (already patched?)")
        return
    with open(path, "w") as f:
        f.write(content.replace(old, new))
    print(f"[OK] {label}")

# Patch 1: triton default_cache_dir 제거됨 (triton 3.x)
with open(SGLANG_UTILS) as f:
    lines = f.readlines()

new_lines = []
i = 0
patched_triton = False
while i < len(lines):
    line = lines[i]
    # triton cache import 블록 찾기
    if "from triton.runtime.cache import" in line and "try:" not in (lines[i-1] if i > 0 else ""):
        # 블록 전체 수집 (멀티라인일 수 있음)
        block = line
        j = i + 1
        while j < len(lines) and ")" not in block.split("#")[0]:
            block += lines[j]
            j += 1
        if j < len(lines):
            block += lines[j]
            j += 1

        indent = len(line) - len(line.lstrip())
        ind = " " * indent

        new_lines.append(f"{ind}try:\n")
        for bl in block.splitlines(keepends=True):
            new_lines.append("    " + bl)
        new_lines.append(f"{ind}except ImportError:\n")
        new_lines.append(f"{ind}    from triton.runtime.cache import FileCacheManager\n")
        new_lines.append(f"{ind}    import os as _os\n")
        new_lines.append(f"{ind}    def default_cache_dir():\n")
        new_lines.append(f'{ind}        return _os.path.join(_os.environ.get("TRITON_CACHE_DIR", _os.path.join(_os.path.expanduser("~"), ".triton", "cache")))\n')
        i = j
        patched_triton = True
        continue
    new_lines.append(line)
    i += 1

if patched_triton:
    with open(SGLANG_UTILS, "w") as f:
        f.writelines(new_lines)
    print("[OK] triton default_cache_dir patch")
else:
    print("[SKIP] triton patch: pattern not found (check manually)")

# Patch 2: vllm._custom_ops (이미 패치됐을 수 있음)
patch_file(
    SGLANG_QUANT,
    "from vllm._custom_ops import scaled_fp8_quant",
    "try:\n    from vllm._custom_ops import scaled_fp8_quant\nexcept Exception:\n    scaled_fp8_quant = None",
    "vllm._custom_ops"
)

# Patch 3: AutoImageProcessor.register() 중복 exist_ok (transformers API 변경)
patch_file(
    os.path.join(_sglang_root, "srt/configs/utils.py"),
    "AutoImageProcessor.register(config, None, image_processor, None, exist_ok=True)",
    "AutoImageProcessor.register(config, image_processor, exist_ok=True)",
    "AutoImageProcessor.register exist_ok"
)

# Patch 4: sglang 전체에서 vllm import를 전부 try/except로 감싸기
import re

def patch_vllm_imports_in_file(filepath):
    with open(filepath) as f:
        content = f.read()

    if "vllm" not in content:
        return False

    lines = content.splitlines(keepends=True)
    new_lines = []
    i = 0
    changed = False

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # 이미 try: 블록 안에 있는지 확인
        prev = lines[i-1].strip() if i > 0 else ""
        if prev in ("try:", ) :
            new_lines.append(line)
            i += 1
            continue

        # "from vllm..." 또는 "import vllm..." 패턴
        if re.match(r'\s*(from vllm\b|import vllm\b)', line):
            indent = len(line) - len(line.lstrip())
            ind = " " * indent

            # 멀티라인 import 수집
            block = line
            while block.rstrip().endswith("\\") or (block.count("(") > block.count(")")):
                i += 1
                if i >= len(lines):
                    break
                block += lines[i]

            # 이미 try로 감싸져 있으면 스킵
            if new_lines and new_lines[-1].strip() == "try:":
                new_lines.append(block)
                i += 1
                changed = True
                continue

            # try/except로 감싸기
            new_lines.append(f"{ind}try:\n")
            for bl in block.splitlines(keepends=True):
                new_lines.append("    " + bl)
            new_lines.append(f"{ind}except Exception:\n")
            new_lines.append(f"{ind}    pass\n")
            changed = True
        else:
            new_lines.append(line)

        i += 1

    if changed:
        with open(filepath, "w") as f:
            f.writelines(new_lines)
    return changed

patched_files = []
for root, dirs, files in os.walk(_sglang_root):
    for fname in files:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(root, fname)
        try:
            if patch_vllm_imports_in_file(fpath):
                patched_files.append(fpath.replace(_sglang_root, ""))
        except Exception as e:
            print(f"[ERR] {fpath}: {e}")

if patched_files:
    print(f"[OK] vllm import patch applied to {len(patched_files)} files:")
    for f in patched_files:
        print(f"     {f}")
else:
    print("[SKIP] vllm import patch: no files needed patching")

# Patch 5: DeepseekVL2Config dataclass 문제 (Python 3.10 + transformers>=4.51)
# vision_config에 default=None 추가해서 non-default-follows-default 에러 수정
deepseekvl2_path = os.path.join(_sglang_root, "srt/configs/deepseekvl2.py")
if os.path.exists(deepseekvl2_path):
    with open(deepseekvl2_path) as f:
        content = f.read()
    # vision_config: VisionEncoderConfig 같은 패턴을 vision_config: VisionEncoderConfig = None으로
    import re as _re
    new_content = _re.sub(
        r'(\s+vision_config:\s*\S+)(\s*\n)',
        r'\1 = None\2',
        content
    )
    if new_content != content:
        with open(deepseekvl2_path, "w") as f:
            f.write(new_content)
        print("[OK] DeepseekVL2Config vision_config default=None patch")
    else:
        print("[SKIP] DeepseekVL2Config patch: pattern not found (already patched?)")
else:
    print("[SKIP] DeepseekVL2Config patch: file not found")

print("\nDone. Run the training script again.")

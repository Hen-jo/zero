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

print("\nDone. Run the training script again.")

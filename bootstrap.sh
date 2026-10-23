#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/smolvla_lab}
BASE_PYTHON=${BASE_PYTHON:-/root/miniconda3/bin/python}
LEROBOT_COMMIT=${LEROBOT_COMMIT:-f9e9d739ea7055b282ea1b165e8785412a674693}
mkdir -p "$PROJECT_ROOT"/{vendor,cache,logs,reports,configs,scripts,runs}
cd "$PROJECT_ROOT"
if [[ ! -d vendor/lerobot/.git ]]; then
  git clone --depth 1 https://github.com/huggingface/lerobot.git vendor/lerobot
fi
if [[ "$(git -C vendor/lerobot rev-parse HEAD)" != "$LEROBOT_COMMIT" ]]; then
  git -C vendor/lerobot diff --quiet
  git -C vendor/lerobot fetch --depth 1 origin "$LEROBOT_COMMIT"
  git -C vendor/lerobot checkout --detach "$LEROBOT_COMMIT"
fi
if [[ ! -x .venv/bin/python ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages .venv
fi
cat > constraints.txt <<'EOF'
torch==2.8.0
torchvision==0.23.0
torchcodec==0.7.0
numpy>=2.0,<2.3
transformers==5.5.4
peft==0.21.2
hf_libero==0.1.4
mujoco==3.8.1
robosuite==1.4.0
accelerate==1.15.0
av==15.1.0
pyarrow==26.0.0
safetensors==0.8.0
huggingface_hub==1.33.0
draccus==0.11.6
EOF
export PIP_NO_CACHE_DIR=1
export UV_CACHE_DIR="$PROJECT_ROOT/cache/uv"
.venv/bin/python -m pip install --disable-pip-version-check uv
.venv/bin/uv pip install --python .venv/bin/python -c constraints.txt \
  -e 'vendor/lerobot[smolvla,peft,libero,training,evaluation]' matplotlib
.venv/bin/python scripts/apply_patches.py
git -C vendor/lerobot rev-parse HEAD > reports/lerobot_commit.txt
.venv/bin/python -m pip freeze > reports/environment.freeze.txt
.venv/bin/python - <<'PY'
import json, platform, torch, transformers, lerobot
from pathlib import Path
report = {
    'python': platform.python_version(), 'torch': torch.__version__,
    'cuda': torch.version.cuda, 'transformers': transformers.__version__,
    'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}
Path('reports/runtime.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
PY

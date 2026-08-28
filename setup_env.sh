#!/usr/bin/env bash
# setup_env.sh -- rebuild the OpenVLA + LIBERO environment from scratch on a wiped box.
#
# Verified working on Brev L40S 48GB (Crusoe), Ubuntu 22.04, 2026-07-26.
# Install ORDER MATTERS: torch -> LIBERO -> safe-openvla -> flash-attn.
#
# Six corrections to the recipe in init_instructions.md Step 0, each found the hard way:
#
#   1. `python3 -m venv` FAILS out of the box -- ensurepip is missing. Needs
#      `apt install python3.10-venv` first.
#   2. LIBERO's top-level `libero/` has no `__init__.py`, so `find_packages()` maps
#      nothing and a modern PEP 660 editable install yields an importable-but-empty
#      namespace package. Install with `--config-settings editable_mode=compat`.
#   3. LIBERO prompts interactively on first import to create ~/.libero/config.yaml.
#      Pre-seed it (or pipe 'n') or any non-tty run hangs/EOFs.
#   4. robosuite 1.4.0 needs the mujoco 2.3.x C API; pip resolves mujoco 3.x, which
#      changed `mj_fullM`'s signature -> TypeError at env construction. Pin mujoco==2.3.2.
#   5. opencv-python 5.x requires numpy>=2, but torch 2.2 / TF 2.15 need numpy<2.
#      Pin opencv-python==4.10.0.84.
#   6. tensorflow-metadata 1.21 needs protobuf>=5.27 (`runtime_version`), which TF 2.15
#      forbids. Pin tensorflow-metadata==1.14.0, which pulls protobuf 3.20.3; that in
#      turn needs wandb<=0.16.x. Pin wandb==0.16.6 to keep `pip check` clean.
#
# Not handled here: nvcc must exist at /usr/local/cuda (env.sh puts it on PATH).

set -euo pipefail

CODE_ROOT=/ephemeral/code
VENV="$CODE_ROOT/venv"
export TMPDIR=/data/tmp
export PIP_CACHE_DIR=/data/tmp/pip-cache

echo "=== 0. preflight ==="
df -h /data | tail -1
AVAIL_GB=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
if [ "$AVAIL_GB" -lt 60 ]; then
    echo "ABORT: /data has ${AVAIL_GB}G free, need >= 60G" >&2
    exit 1
fi
mkdir -p /data/tmp /data/hf-cache /data/corpus

echo "=== 1. system deps ==="
# libosmesa6-dev: required for MuJoCo headless. python3.10-venv: see correction 1.
sudo apt-get update -qq
sudo apt-get install -y libosmesa6-dev python3.10-venv python3-dev

echo "=== 2. repos ==="
cd "$CODE_ROOT"
# The user's fork, not upstream vla-safe/openvla.
[ -d openvla ] || git clone git@github.com:James-blissett/openvla.git openvla
git -C openvla remote get-url upstream >/dev/null 2>&1 || \
    git -C openvla remote add upstream https://github.com/vla-safe/openvla.git
[ -d LIBERO ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

echo "=== 3. venv + torch (cu121) ==="
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip setuptools wheel
pip install -q torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== 4. LIBERO (before safe-openvla) ==="
pip install -q -r openvla/experiments/robot/libero/libero_requirements.txt
# correction 2: legacy-compat editable mode
pip install -q -e ./LIBERO --config-settings editable_mode=compat
# correction 3: pre-seed the config LIBERO would otherwise prompt for
if [ ! -f "$HOME/.libero/config.yaml" ]; then
    mkdir -p "$HOME/.libero"
    L="$CODE_ROOT/LIBERO/libero/libero"
    cat > "$HOME/.libero/config.yaml" <<EOF
assets: $L/./assets
bddl_files: $L/./bddl_files
benchmark_root: $L
datasets: $L/../datasets
init_states: $L/./init_files
EOF
fi

echo "=== 5. safe-openvla (respects its transformers/timm pins) ==="
pip install -q -e ./openvla

echo "=== 6. dependency corrections (4, 5, 6) ==="
pip install -q "mujoco==2.3.2" "opencv-python==4.10.0.84" \
               "tensorflow-metadata==1.14.0" "wandb==0.16.6" "numpy<2"

echo "=== 7. flash-attn 2.5.8 (20-40 min; run under tmux) ==="
# Needs nvcc on PATH + CUDA_HOME, which env.sh exports.
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
pip install flash-attn==2.5.8 --no-build-isolation --no-cache-dir

echo "=== 8. verify ==="
pip check
python - <<'PY'
import torch, transformers, timm, tokenizers, numpy, flash_attn, mujoco, robosuite
print("torch       ", torch.__version__, "| cuda:", torch.cuda.is_available())
print("transformers", transformers.__version__, "(expect 4.40.1)")
print("timm        ", timm.__version__, "(expect 0.9.10)")
print("tokenizers  ", tokenizers.__version__, "(expect 0.19.1)")
print("numpy       ", numpy.__version__, "(expect <2)")
print("flash_attn  ", flash_attn.__version__, "(expect 2.5.8)")
print("mujoco      ", mujoco.__version__, "(expect 2.3.2)")
print("robosuite   ", robosuite.__version__, "(expect 1.4.0)")
from libero.libero import benchmark
print("libero_10 tasks:", benchmark.get_benchmark_dict()["libero_10"]().n_tasks, "(expect 10)")
PY

echo
echo "=== setup complete. next: source env.sh && python collect/gen_rollouts.py --smoke ==="

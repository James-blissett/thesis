# Source before any run: source env.sh
export MUJOCO_GL=egl
export HF_HOME=/data/hf-cache
export TMPDIR=/data/tmp

# CUDA toolkit is installed but not on PATH by default; the flash-attn build needs both.
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export PIP_CACHE_DIR=/data/tmp/pip-cache

# Project venv (Python 3.10), shared by the openvla fork, LIBERO, and thesis scripts.
if [ -f /ephemeral/code/venv/bin/activate ]; then
    source /ephemeral/code/venv/bin/activate
fi

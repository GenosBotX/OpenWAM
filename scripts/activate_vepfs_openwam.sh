# Source this file in bash to use the shared environment.
export OPENWAM_REPO_ROOT=/vePFS/tangyuting/OpenWAM
export CONDA_PREFIX=/vePFS/tangyuting/envs/openwam
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$OPENWAM_REPO_ROOT:${PYTHONPATH:-}"
export HF_HOME=/vePFS/tangyuting/cache/openwam/huggingface
export WANDB_MODE=${WANDB_MODE:-offline}
cd "$OPENWAM_REPO_ROOT"

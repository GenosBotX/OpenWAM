#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${OPENWAM_PYTHON:-/vePFS/tangyuting/envs/openwam/bin/python}"
export OPENWAM_REPO_ROOT="$repo_root"
export PYTHONPATH="$repo_root:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/vePFS/tangyuting/cache/openwam/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/openwam-torch-extensions}"
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
if [[ "${NNODES:-1}" != 1 ]]; then
  echo "本配置需要单节点 8 卡" >&2
  exit 2
fi
cd "$repo_root"
config_name=train_piper_x_20261022_20261023_tos_eef_8gpu_e10
launch=("$python_bin" scripts/run_with_tos_credentials.py "${OPENWAM_TOS_CREDENTIALS:-${TOS_CREDENTIAL_FILE:-/vePFS/tangyuting/.config/openwam/tos-linglong.json}}"
  env "TOS_ENDPOINT=${TOS_ENDPOINT:-tos-cn-beijing.ivolces.com}"
  "$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8
  scripts/train.py "--config-name=$config_name" "$@")
if [[ "${OPENWAM_LAUNCH_DRYRUN:-0}" == 1 ]]; then
  printf '%q ' "${launch[@]}"
  echo
  exit 0
fi
exec "${launch[@]}"

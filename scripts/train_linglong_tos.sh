#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
: "${LINGLONG_TOS_DATASET:?Set LINGLONG_TOS_DATASET to a published tos:// bucket prefix}"
if [[ "$LINGLONG_TOS_DATASET" != tos://* ]]; then
  echo 'LINGLONG_TOS_DATASET must start with tos://' >&2
  exit 2
fi
if [[ -n "${OPENWAM_LINGLONG_DATASET_DIR:-}" && "$OPENWAM_LINGLONG_DATASET_DIR" != "$LINGLONG_TOS_DATASET" ]]; then
  echo 'Conflicting OPENWAM_LINGLONG_DATASET_DIR override' >&2
  exit 2
fi
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HOME="${HF_HOME:-/vePFS/tangyuting/cache/openwam/huggingface}"
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/openwam-torch-extensions}"
python_bin="${OPENWAM_PYTHON:-/vePFS/tangyuting/envs/openwam/bin/python}"
config="${OPENWAM_TOS_CONFIG:-train_linglong_tos}"
launch=("$python_bin" -m torch.distributed.run --standalone --nnodes=1
  "--nproc_per_node=${NPROC_PER_NODE:-8}" --module scripts.train "--config-name=$config" "$@")
if [[ -n "${TOS_CREDENTIAL_FILE:-}" ]]; then
  exec "$python_bin" scripts/run_with_tos_credentials.py "$TOS_CREDENTIAL_FILE" "${launch[@]}"
fi
exec "${launch[@]}"

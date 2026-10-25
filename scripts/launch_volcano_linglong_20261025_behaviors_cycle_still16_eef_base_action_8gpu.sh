#!/usr/bin/env bash
set -euo pipefail
export OPENWAM_REPO_ROOT=/vePFS/zhuoyu/OpenWAM
export CONDA_PREFIX=/vePFS/tangyuting/envs/openwam
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$OPENWAM_REPO_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/vePFS/tangyuting/cache/openwam/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
cd "$OPENWAM_REPO_ROOT"
PYTHON_BIN="$CONDA_PREFIX/bin/python"
CONFIG_NAME=train_linglong_20261025_behaviors_cycle_still16_eef_base_action_8gpu
DATASET_DIR=/vePFS/data/linglong/linglong_20261025_behaviors_cycle_still16_openwam/linglong_openwam_train_15hz
OUTPUT_DIR=/vePFS/zhuoyu/OpenWAM/outputs/linglong_20261025_behaviors_cycle_still16_eef_base_action_8gpu_e10
# One worker, eight GPUs. Local cycle-prompt data, EEF proprio, chassis action supervised.
if [[ "${NNODES:-1}" != 1 ]]; then
  echo "ERROR: this recipe requires one worker with eight GPUs" >&2
  exit 2
fi
role="${MLP_ROLE:-worker}"
upper_role="$(printf '%s' "$role" | tr '[:lower:]' '[:upper:]')"
hosts_var="MLP_${upper_role}_ALL_HOSTS"
if [[ ! "$hosts_var" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
  echo "ERROR: invalid MLP_ROLE" >&2
  exit 2
fi
hosts="${!hosts_var:-${MLP_WORKER_ALL_HOSTS:-}}"
hosts="${hosts//,/ }"
host_list=( $hosts )
if (( ${#host_list[@]} > 1 )); then
  echo "ERROR: select one worker with eight GPUs for this recipe" >&2
  exit 2
fi
unset OPENWAM_LINGLONG_SOURCE_ROOT OPENWAM_LINGLONG_DATASET_DIR
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/openwam-torch-extensions}"
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
if [[ "${OPENWAM_LAUNCH_DRYRUN:-0}" != 1 ]]; then
  mkdir -p "$OPENWAM_REPO_ROOT/logs/linglong_20261025_behaviors_cycle_still16_eef_base_action"
  log_path="$OPENWAM_REPO_ROOT/logs/linglong_20261025_behaviors_cycle_still16_eef_base_action/launch_$(date +%Y%m%d_%H%M%S)_$(hostname)_$$.log"
  exec > >(tee -a "$log_path") 2>&1
  echo "OpenWAM cycle still16 EEF+base-action launch log: $log_path"
fi
for artifact in manifest.json meta/normalization_stats.json meta/state_eef_stats.json meta/action_eef_stats.json data/chunk-000/episode_000000.parquet; do
  [[ -s "$DATASET_DIR/$artifact" ]] || { echo "ERROR: missing data asset: $DATASET_DIR/$artifact" >&2; exit 2; }
done
if [[ ! -d "$DATASET_DIR/images" ]]; then
  echo "ERROR: image root is unavailable: $DATASET_DIR/images" >&2
  exit 2
fi
for artifact in "$OPENWAM_REPO_ROOT/configs/${CONFIG_NAME}.yaml" "$OPENWAM_REPO_ROOT/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model/config.yaml" "$OPENWAM_REPO_ROOT/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors" /vePFS/tangyuting/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth; do
  [[ -s "$artifact" ]] || { echo "ERROR: missing model asset: $artifact" >&2; exit 2; }
done
launch_command=("$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/train.py "--config-name=$CONFIG_NAME" "$@")
echo "OpenWAM 20261025 cycle still16: EEF proprio, chassis action supervised; one worker x eight GPUs; batch=6 accumulation=24 epochs=10"
echo "dataset: $DATASET_DIR"
echo "checkpoints: $OUTPUT_DIR"
if [[ "${OPENWAM_LAUNCH_DRYRUN:-0}" == 1 ]]; then
  printf '%q ' "${launch_command[@]}"
  echo
  exit 0
fi
exec "${launch_command[@]}"

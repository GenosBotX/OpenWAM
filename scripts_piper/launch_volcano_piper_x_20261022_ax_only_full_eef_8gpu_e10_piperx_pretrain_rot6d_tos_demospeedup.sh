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
CONFIG_NAME=train_piper_x_20261022_ax_only_full_eef_8gpu_e10_piperx_pretrain_rot6d_tos_demospeedup
PRETRAIN=/vePFS/data/openwam/OpenWAM-Alpha-Real-RoboDojo-Piper-X
ENTROPY_DIR=/vePFS/zhuoyu/OpenWAM/outputs/action_entropy_piper_x_20261022_ax_only_full_step18560_stride2
export TOS_CREDENTIAL_FILE="${TOS_CREDENTIAL_FILE:-/vePFS/tangyuting/.config/openwam/tos-linglong.json}"
# One worker, eight GPUs. Same rot6d recipe as the local 1022 run:
# Piper-X step 30000, batch 4, accumulation 24, 10 epochs.
# Data and 20D statistics both come from the 1022 TOS train prefix.
# DemoSpeedup keep-tables use the finished stride-2 entropy files.
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
  mkdir -p "$OPENWAM_REPO_ROOT/logs/piper_x_20261022_rot6d_tos_demospeedup"
  log_path="$OPENWAM_REPO_ROOT/logs/piper_x_20261022_rot6d_tos_demospeedup/launch_$(date +%Y%m%d_%H%M%S)_$(hostname)_$$.log"
  exec > >(tee -a "$log_path") 2>&1
  echo "OpenWAM Piper X 1022 TOS DemoSpeedup launch log: $log_path"
fi
entropy_count="$(find "$ENTROPY_DIR" -maxdepth 1 -type f -name 'episode_*.npz' | wc -l)"
if [[ "$entropy_count" != 49 ]]; then
  echo "ERROR: expected 49 entropy files in $ENTROPY_DIR, found $entropy_count" >&2
  exit 2
fi
[[ -f "$TOS_CREDENTIAL_FILE" ]] || { echo "ERROR: missing TOS credential file" >&2; exit 2; }
for artifact in "$OPENWAM_REPO_ROOT/configs/${CONFIG_NAME}.yaml" "$PRETRAIN/config.yaml" "$PRETRAIN/checkpoint_step_30000.safetensors" /vePFS/tangyuting/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth; do
  [[ -s "$artifact" ]] || { echo "ERROR: missing model asset: $artifact" >&2; exit 2; }
done
train=(
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8
  scripts/train.py "--config-name=$CONFIG_NAME" "$@"
)
launch_command=("$PYTHON_BIN" scripts/run_with_tos_credentials.py "$TOS_CREDENTIAL_FILE" "${train[@]}")
echo "OpenWAM Piper X 20261022 TOS: Piper-X step 30000; rot6d; 1022-only stats; DemoSpeedup r=2/3; eight GPUs; batch=4 accumulation=24 epochs=10"
echo "checkpoints: $OPENWAM_REPO_ROOT/outputs/piper_x_20261022_ax_only_full_eef_8gpu_e10_piperx_pretrain_rot6d_tos_demospeedup"
if [[ "${OPENWAM_LAUNCH_DRYRUN:-0}" == 1 ]]; then
  printf '%q ' "${launch_command[@]}"
  echo
  exit 0
fi
exec "${launch_command[@]}"

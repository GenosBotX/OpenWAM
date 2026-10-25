#!/usr/bin/env bash
# One worker, eight GPUs. Action entropy for the 2026-10-23 pickup model.
# Stride 2 and 8 noises match the fittings entropy used by DemoSpeedup.
# Train and test episode indices both start at 0, so the test npz files go
# in a separate directory. The test run starts after the train run finishes.
set -euo pipefail
export OPENWAM_REPO_ROOT=/vePFS/zhuoyu/OpenWAM
export CONDA_PREFIX=/vePFS/tangyuting/envs/openwam
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$OPENWAM_REPO_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/vePFS/tangyuting/cache/openwam/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
cd "$OPENWAM_REPO_ROOT"
PYTHON_BIN="$CONDA_PREFIX/bin/python"
CKPT_DIR=/vePFS/zhuoyu/OpenWAM/outputs/linglong_1023_eef_8gpu_bs6_e10/2026-10-23_17-49-26
CKPT_NAME=checkpoint_step_2230.safetensors
TRAIN_DIR=/vePFS/data/linglong/linglong_1023_openwam/linglong_openwam_train_15hz
TEST_DIR=/vePFS/data/linglong/linglong_1023_openwam/linglong_openwam_test_15hz
FRAME_STRIDE=2
TRAIN_OUTPUT="$OPENWAM_REPO_ROOT/outputs/action_entropy_linglong_1023_eef_step2230_stride2"
TEST_OUTPUT="$OPENWAM_REPO_ROOT/outputs/action_entropy_linglong_1023_eef_step2230_stride2_test"
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
for artifact in \
  "$TRAIN_DIR/manifest.json" \
  "$TRAIN_DIR/meta/episodes.jsonl" \
  "$TRAIN_DIR/meta/state_eef_stats.json" \
  "$TRAIN_DIR/meta/action_eef_stats.json" \
  "$TEST_DIR/manifest.json" \
  "$TEST_DIR/meta/episodes.jsonl" \
  "$TEST_DIR/meta/state_eef_stats.json" \
  "$TEST_DIR/meta/action_eef_stats.json" \
  "$CKPT_DIR/config.yaml" \
  "$CKPT_DIR/$CKPT_NAME"
do
  if [[ ! -f "$artifact" ]]; then
    echo "ERROR: missing artifact: $artifact" >&2
    exit 2
  fi
done
if [[ "${OPENWAM_LAUNCH_DRYRUN:-0}" != 1 ]]; then
  mkdir -p "$OPENWAM_REPO_ROOT/logs/action_entropy_linglong_1023_step2230" "$TRAIN_OUTPUT" "$TEST_OUTPUT"
  log_path="$OPENWAM_REPO_ROOT/logs/action_entropy_linglong_1023_step2230/launch_$(date +%Y%m%d_%H%M%S)_$(hostname)_$$.log"
  exec > >(tee -a "$log_path") 2>&1
  echo "OpenWAM 1023 action-entropy log: $log_path"
fi
echo "checkpoint=$CKPT_DIR/$CKPT_NAME"
echo "frame_stride=$FRAME_STRIDE noises=8"
echo "train=$TRAIN_DIR -> $TRAIN_OUTPUT"
echo "test=$TEST_DIR -> $TEST_OUTPUT"
run_split() {
  local dataset_dir="$1"
  local output_dir="$2"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/run_linglong_action_entropy.py \
    --checkpoint-dir "$CKPT_DIR" \
    --checkpoint-name "$CKPT_NAME" \
    --dataset-dir "$dataset_dir" \
    --output-dir "$output_dir" \
    --frame-stride "$FRAME_STRIDE" \
    --noises 8
}
if [[ "${OPENWAM_LAUNCH_DRYRUN:-0}" == 1 ]]; then
  echo "dry-run train then test"
  exit 0
fi
run_split "$TRAIN_DIR" "$TRAIN_OUTPUT"
run_split "$TEST_DIR" "$TEST_OUTPUT"

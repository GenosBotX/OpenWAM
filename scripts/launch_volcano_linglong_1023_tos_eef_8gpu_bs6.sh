#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# 2026-10-23 sheet pickup. One prompt, no left/right split.
# Train and test are separate immutable TOS prefixes.
export LINGLONG_TOS_DATASET="${LINGLONG_TOS_DATASET:-tos://ml-platform-auto-created-required-2109640594-cn-beijing/linglong/openwam/linglong/20261024_nabangzi_1023_eef_v1}"
export LINGLONG_TOS_EVAL_DATASET="${LINGLONG_TOS_EVAL_DATASET:-tos://ml-platform-auto-created-required-2109640594-cn-beijing/linglong/openwam/linglong/20261024_nabangzi_1023_eef_v1_test}"
export TOS_CREDENTIAL_FILE="${TOS_CREDENTIAL_FILE:-/vePFS/tangyuting/.config/openwam/tos-linglong.json}"
export OPENWAM_PYTHON="${OPENWAM_PYTHON:-/vePFS/tangyuting/envs/openwam/bin/python}"
export OPENWAM_TOS_CONFIG=train_linglong_tos
export NPROC_PER_NODE=8
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
unset OPENWAM_LINGLONG_DATASET_DIR OPENWAM_LINGLONG_SOURCE_ROOT

output_path="${LINGLONG_TOS_OUTPUT_PATH:-/vePFS/zhuoyu/OpenWAM/outputs/linglong_1023_tos_eef_8gpu_bs6_acc24}"
launch=(bash scripts/train_linglong_tos.sh
  training.batch_size=6
  training.gradient_accumulation_steps=24
  dataloader.window_stride=1
  training.num_epochs=5
  training.resume_ckpt_path=null
  "training.output_path=$output_path"
  "project.output_dir=$output_path/hydra"
  "dataloader.eval_dataset_dir=$LINGLONG_TOS_EVAL_DATASET"
  "$@")

echo "Linglong 1023 TOS EEF: 1 node x 8 GPUs, batch 6/rank, accumulation 24, window stride 1, 5 epochs"
echo "Dataset: $LINGLONG_TOS_DATASET"
echo "Eval:    $LINGLONG_TOS_EVAL_DATASET"
echo "Output:  $output_path"
if [[ "${OPENWAM_LAUNCH_DRYRUN:-0}" == 1 ]]; then
  printf '%q ' "${launch[@]}"
  printf '\n'
  exit 0
fi
[[ -x "$OPENWAM_PYTHON" ]] || { echo "ERROR: missing Python: $OPENWAM_PYTHON" >&2; exit 2; }
[[ -r "$TOS_CREDENTIAL_FILE" ]] || { echo "ERROR: missing credential file: $TOS_CREDENTIAL_FILE" >&2; exit 2; }
exec "${launch[@]}"

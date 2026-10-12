#!/usr/bin/env bash
set -euo pipefail

# Same model/optimizer configuration as 2026-09-27_18-28-59, with the
# merged dataset/output paths and full-state resume saving enabled.
# Additional arguments are passed through as Hydra overrides.
REPO_ROOT="/iag_ad_01/ad/xiejinxia/code/test/OpenWAM"
PYTHON_BIN="${PYTHON_BIN:-/iag_ad_01/ad/xiejinxia/envs/dreamzero/bin/python}"
NPROC="${NPROC:-8}"
EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE:-32}"
MASTER_PORT="${MASTER_PORT:-29500}"
DATASET_DIR="$REPO_ROOT/data/linglong_0923_1010_openwam_train_15hz"

cd "$REPO_ROOT"
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ || ! "$EXPECTED_WORLD_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NPROC and EXPECTED_WORLD_SIZE must be positive integers" >&2
  exit 2
fi
role="${MLP_ROLE:-worker}"
upper_role="$(printf '%s' "$role" | tr '[:lower:]' '[:upper:]')"
hosts_var="MLP_${upper_role}_ALL_HOSTS"
if [[ ! "$hosts_var" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
  echo "ERROR: invalid MLP_ROLE" >&2
  exit 2
fi
ALL_HOSTS_STR="${!hosts_var:-${MLP_WORKER_ALL_HOSTS:-}}"
ALL_HOSTS_STR="${ALL_HOSTS_STR//,/ }"
ALL_HOSTS_STR="${ALL_HOSTS_STR//$'\n'/ }"
if [[ -z "$ALL_HOSTS_STR" ]]; then
  if [[ "${NNODES:-}" == 1 ]]; then
    ALL_HOSTS_STR="$(hostname)"
  else
    echo "ERROR: cannot find worker host list; set NNODES=1 for standalone training" >&2
    exit 2
  fi
fi
read -r -a hosts <<< "$ALL_HOSTS_STR"
if (( ${#hosts[@]} == 0 )); then
  echo "ERROR: worker host list is empty" >&2
  exit 2
fi
if [[ -n "${NNODES:-}" && "$NNODES" != "${#hosts[@]}" ]]; then
  echo "ERROR: NNODES=$NNODES disagrees with host count ${#hosts[@]}" >&2
  exit 2
fi
NNODES="${#hosts[@]}"
MASTER_ADDR="${MASTER_ADDR:-${hosts[0]}}"
if [[ "$((NNODES * NPROC))" != "$EXPECTED_WORLD_SIZE" ]]; then
  echo "ERROR: expected WORLD_SIZE=$EXPECTED_WORLD_SIZE, got NNODES=$NNODES NPROC=$NPROC" >&2
  exit 2
fi
if [[ -z "${NODE_RANK:-}" ]]; then
  THIS_HOST="$(hostname)"
  THIS_IPS="$(hostname -I 2>/dev/null || true)"
  for i in "${!hosts[@]}"; do
    if [[ "${hosts[$i]}" == "$THIS_HOST" || " $THIS_IPS " == *" ${hosts[$i]} "* ]]; then
      NODE_RANK="$i"
      break
    fi
  done
  if [[ -z "${NODE_RANK:-}" && "$THIS_HOST" =~ worker-([0-9]+)$ ]]; then
    NODE_RANK="$((10#${BASH_REMATCH[1]}))"
  fi
fi
if [[ "$NNODES" == 1 && -z "${NODE_RANK:-}" ]]; then
  NODE_RANK=0
fi
if [[ ! "${NODE_RANK:-}" =~ ^(0|[1-9][0-9]*)$ ]] || (( NODE_RANK >= NNODES )); then
  echo "ERROR: cannot resolve a valid NODE_RANK in [0, $((NNODES-1))]" >&2
  exit 2
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
# This merged bucket resolves both collection-specific image roots itself.
unset OPENWAM_LINGLONG_SOURCE_ROOT
unset OPENWAM_LINGLONG_DATASET_DIR
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

echo "OpenWAM Linglong merged launch: NNODES=$NNODES NPROC=$NPROC WORLD_SIZE=$((NNODES*NPROC))"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NODE_RANK=$NODE_RANK"
if [[ ! -f "$DATASET_DIR/manifest.json" || ! -f "$DATASET_DIR/meta/normalization_stats.npy" ]]; then
  echo "ERROR: merged Linglong bucket is not ready: $DATASET_DIR" >&2
  exit 2
fi
if [[ ! -f "$REPO_ROOT/configs/train_linglong_0923_1010.yaml" ]]; then
  echo "ERROR: merged training config is missing" >&2
  exit 2
fi

launch_command=("$PYTHON_BIN" -m torch.distributed.run \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  scripts/train.py \
  --config-name=train_linglong_0923_1010 "$@")
if [[ "${OPENWAM_LAUNCH_DRYRUN:-0}" == 1 ]]; then
  printf '%q ' "${launch_command[@]}"
  printf '\n'
  exit 0
fi
exec "${launch_command[@]}"

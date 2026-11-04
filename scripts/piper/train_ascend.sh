#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
task_root="$(dirname "$repo_root")"
export OPENWAM_REPO_ROOT="$repo_root"
export OPENWAM_DATA_ROOT="${OPENWAM_DATA_ROOT:-obs://genosbot-shanghai/openwam/data/piper-x}"
export OPENWAM_OBS_ENDPOINT="${OPENWAM_OBS_ENDPOINT:-https://obs.cn-southwest-2.myhuaweicloud.com}"
export OPENWAM_CACHE_DIR="${OPENWAM_CACHE_DIR:-/tmp/openwam-piper-obs-cache}"
export OPENWAM_CACHE_MAX_BYTES="${OPENWAM_CACHE_MAX_BYTES:-4294967296}"
export OPENWAM_CACHE_MIN_FREE_BYTES="${OPENWAM_CACHE_MIN_FREE_BYTES:-1073741824}"
export OPENWAM_BASE_CHECKPOINT="${OPENWAM_BASE_CHECKPOINT:-$task_root/models/OpenWAM-Alpha-Real-RoboDojo-Piper-X/checkpoint_step_30000.safetensors}"
export OPENWAM_WAN_MODEL_DIR="${OPENWAM_WAN_MODEL_DIR:-/home/ma-user/work/customer-test/OpenWAM-main/assets/video_backbone_ckpt/Wan2.2-TI2V-5B}"
# Reuse the runtime of customer-test/openwam/epoch_foundation.sh.
ascend_env="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ ! -f "$ascend_env" ]]; then
  echo "Ascend environment script not found: $ascend_env" >&2
  exit 2
fi
set +u
source "$ascend_env"
set -u
python_bin="${OPENWAM_PYTHON:-/home/ma-user/anaconda3/envs/openwam/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "OpenWAM Python not found: $python_bin; use the existing OpenWAM training image or set OPENWAM_PYTHON" >&2
  exit 2
fi
export PATH="$(dirname "$python_bin"):$PATH"
export PYTHONPATH="$repo_root:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export HCCL_WHITELIST_DISABLE="${HCCL_WHITELIST_DISABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# ModelArts exposes the worker count and rank through VC/MA variables.  Keep
# NNODES/NPROC_PER_NODE/NODE_RANK overrides for local smoke tests and reruns.
nnodes="${NNODES:-${MA_NUM_HOSTS:-1}}"
if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  nproc="$NPROC_PER_NODE"
else
  # ModelArts MA_NUM_GPUS and npu-smi info -l can report the job wrapper
  # count rather than visible devices. Huawei Piper-X jobs use 8 NPUs/node;
  # override this for a different resource shape.
  nproc="${OPENWAM_NPROC_PER_NODE:-8}"
fi
if [[ "$nnodes" != 1 && -z "${NODE_RANK:-${VC_TASK_INDEX:-}}" ]]; then
  echo "Multi-node launch needs NODE_RANK or ModelArts VC_TASK_INDEX" >&2
  exit 2
fi
node_rank="${NODE_RANK:-${VC_TASK_INDEX:-0}}"
master_addr="${MASTER_ADDR:-}"
if [[ -z "$master_addr" && -n "${VC_WORKER_HOSTS:-}" ]]; then
  master_addr="$("$python_bin" - <<'PY'
import json
import os
import re
raw = os.environ["VC_WORKER_HOSTS"].strip()
hosts = json.loads(raw) if raw.startswith("[") else re.split(r"[,;\s]+", raw)
if not isinstance(hosts, list) or not hosts or not isinstance(hosts[0], str):
    raise SystemExit("Invalid VC_WORKER_HOSTS: expected a host list")
host = hosts[0].strip()
if host.count(":") == 1:
    host, port = host.rsplit(":", 1)
    if not port.isdigit():
        raise SystemExit("Invalid VC_WORKER_HOSTS host:port")
if not host or re.search(r"[\s\[\]\"']", host):
    raise SystemExit("Invalid master hostname in VC_WORKER_HOSTS")
print(host)
PY
)"
fi
if [[ "$nnodes" != 1 && -z "$master_addr" ]]; then
  echo "Multi-node launch needs MASTER_ADDR or ModelArts VC_WORKER_HOSTS" >&2
  exit 2
fi
master_addr="${master_addr:-127.0.0.1}"
if [[ "$nnodes" != 1 && ( "$master_addr" == 127.* || "$master_addr" == localhost || "$master_addr" == ::1 ) ]]; then
  echo "Multi-node MASTER_ADDR cannot be a loopback address" >&2
  exit 2
fi
if ! [[ "$nnodes" =~ ^[1-9][0-9]*$ && "$nproc" =~ ^[1-9][0-9]*$ && "$node_rank" =~ ^[0-9]+$ ]]; then
  echo "Invalid distributed settings: NNODES=$nnodes NPROC_PER_NODE=$nproc NODE_RANK=$node_rank" >&2
  exit 2
fi
if (( node_rank >= nnodes )); then
  echo "NODE_RANK=$node_rank must be smaller than NNODES=$nnodes" >&2
  exit 2
fi
batch="${BATCH_SIZE:-6}"
effective="${EFFECTIVE_BATCH:-768}"
if ! [[ "$batch" =~ ^[1-9][0-9]*$ && "$effective" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE and EFFECTIVE_BATCH must be positive integers" >&2
  exit 2
fi
if (( effective % (nnodes * nproc * batch) != 0 )); then
  echo "EFFECTIVE_BATCH must divide evenly across nodes, devices and batch size" >&2
  exit 2
fi
accum="$((effective / (nnodes * nproc * batch)))"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 2
  fi
}
require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Required directory is missing: $1" >&2
    exit 2
  fi
}

require_file "$python_bin"
require_dir "$OPENWAM_REPO_ROOT"
require_file "$OPENWAM_BASE_CHECKPOINT"
require_file "${OPENWAM_BASE_CHECKPOINT%/*}/config.yaml"
require_dir "${OPENWAM_BASE_CHECKPOINT%/*}/tokenizer/google/umt5-xxl"
require_dir "$OPENWAM_WAN_MODEL_DIR"
require_file "$OPENWAM_WAN_MODEL_DIR/diffusion_pytorch_model.safetensors.index.json"
require_file "$OPENWAM_WAN_MODEL_DIR/Wan2.2_VAE.pth"
# Use the training reader for both local overrides and OBS access checks.
# This checks every split and shared statistics without mirroring the dataset.
"$python_bin" -m openwam.dataloader.piper_obs_preflight

if [[ "${OPENWAM_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "Piper-X Ascend preflight passed"
  echo "repo=$OPENWAM_REPO_ROOT"
  echo "python=$python_bin"
  echo "data=$OPENWAM_DATA_ROOT"
  echo "foundation=$OPENWAM_BASE_CHECKPOINT"
  echo "wan=$OPENWAM_WAN_MODEL_DIR"
  echo "distributed=$nnodes nodes x $nproc devices, rank=$node_rank, master=$master_addr:${MASTER_PORT:-29500}"
  exit 0
fi

cd "$repo_root"
launch=("$python_bin" -m torch.distributed.run
  --nnodes="$nnodes" --nproc_per_node="$nproc"
  --node_rank="$node_rank" --master_addr="$master_addr"
  --master_port="${MASTER_PORT:-29500}" scripts/train.py
  --config-name=train_piper_x_ascend
  "training.batch_size=$batch" "training.gradient_accumulation_steps=$accum" "$@")
echo "Piper-X Ascend: $nnodes nodes x $nproc devices; batch=$batch accumulation=$accum effective_batch=$effective"
if [[ "${OPENWAM_LAUNCH_DRYRUN:-0}" == 1 ]]; then
  printf '%q ' "${launch[@]}"
  printf '\n'
  exit 0
fi
exec "${launch[@]}"

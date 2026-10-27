#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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

if [[ "${NNODES:-2}" != 2 || "${NPROC_PER_NODE:-8}" != 8 ]]; then
  echo "本配置需要 2 节点，每节点 8 卡" >&2
  exit 2
fi
role="${MLP_ROLE:-worker}"
upper_role="$(printf '%s' "$role" | tr '[:lower:]' '[:upper:]')"
hosts_var="MLP_${upper_role}_ALL_HOSTS"
if [[ ! "$hosts_var" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
  echo "MLP_ROLE 无效" >&2
  exit 2
fi
hosts="${!hosts_var:-${MLP_WORKER_ALL_HOSTS:-}}"
hosts="${hosts//,/ }"
hosts="${hosts//$'\n'/ }"
read -r -a host_list <<< "$hosts"
if [[ -n "$hosts" && "${#host_list[@]}" != 2 ]]; then
  echo "需要恰好 2 个 worker host" >&2
  exit 2
fi
if [[ -z "${NODE_RANK:-}" ]]; then
  this_host="$(hostname)"
  this_ips="$(hostname -I 2>/dev/null || true)"
  for i in "${!host_list[@]}"; do
    if [[ "${host_list[$i]}" == "$this_host" || " $this_ips " == *" ${host_list[$i]} "* ]]; then
      NODE_RANK="$i"
      break
    fi
  done
  if [[ -z "${NODE_RANK:-}" && "$this_host" =~ worker-([0-9]+)$ ]]; then
    NODE_RANK="$((10#${BASH_REMATCH[1]}))"
  fi
fi
if [[ ! "${NODE_RANK:-}" =~ ^[01]$ ]]; then
  echo "无法确定 NODE_RANK，请设置为 0 或 1" >&2
  exit 2
fi
export NNODES=2 NPROC_PER_NODE=8 NODE_RANK
export MASTER_ADDR="${MASTER_ADDR:-${host_list[0]:-}}"
export MASTER_PORT="${MASTER_PORT:-29500}"
if [[ -z "$MASTER_ADDR" ]]; then
  echo "请提供 Volcano worker hosts 或 MASTER_ADDR" >&2
  exit 2
fi
cd "$repo_root"
config_name=train_piper_x_20261022_20261026_tos_eef_16gpu_e5
launch=("$python_bin" scripts/run_with_tos_credentials.py
  "${OPENWAM_TOS_CREDENTIALS:-${TOS_CREDENTIAL_FILE:-/vePFS/tangyuting/.config/openwam/tos-linglong.json}}"
  env "TOS_ENDPOINT=${TOS_ENDPOINT:-tos-cn-beijing.ivolces.com}"
  "$python_bin" -m torch.distributed.run --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE"
  --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
  scripts/train.py "--config-name=$config_name" "$@")
if [[ "${OPENWAM_LAUNCH_DRYRUN:-0}" == 1 ]]; then
  printf '%q ' "${launch[@]}"
  printf '\n'
  exit 0
fi
exec "${launch[@]}"

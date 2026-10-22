#!/usr/bin/env bash
# 后台启动；已有服务运行时直接返回。配置可通过同名环境变量覆盖。
set -euo pipefail
PIPER_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PIPER_ENV=${PIPER_ENV:-/vePFS/tangyuting/envs/openwam}
PIPER_MODEL=${PIPER_MODEL:-/vePFS/data/openwam/OpenWAM-Alpha-Real-RoboDojo-Piper-X}
PIPER_CKPT=${PIPER_CKPT:-checkpoint_step_30000.safetensors}
PIPER_HOST=${PIPER_HOST:-127.0.0.1}
PIPER_PORT=${PIPER_PORT:-8848}
PIPER_DEVICE=${PIPER_DEVICE:-cuda:0}
PIPER_LOG=${PIPER_LOG:-$PIPER_REPO/logs/piper_x_ws}
export PATH="$PIPER_ENV/bin:$PATH"
export PYTHONPATH="$PIPER_REPO:$PIPER_REPO/third_party:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/vePFS/tangyuting/cache/openwam/huggingface}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$PIPER_REPO"
mkdir -p "$PIPER_LOG"
already_running() {
    if [[ -f "$PIPER_LOG/server.pid" ]]; then
        PIPER_PID=$(cat "$PIPER_LOG/server.pid")
        if [[ "$PIPER_PID" =~ ^[0-9]+$ ]] && kill -0 "$PIPER_PID" 2>/dev/null &&
            grep -aq 'scripts/serve_piper_x.py' "/proc/$PIPER_PID/cmdline"; then
            echo "Piper-X 正在加载或运行：PID=$PIPER_PID"
            return 0
        fi
    fi
    return 1
}
# 先识别已有进程，也兼容旧启动脚本将锁继承给服务进程的情况。
already_running && exit 0
exec 9>"$PIPER_LOG/launch.lock"
flock -n 9 || { echo '另一个 Piper-X 启动操作正在进行'; exit 1; }
already_running && exit 0
"$PIPER_ENV/bin/python" - "$PIPER_HOST" "$PIPER_PORT" <<'PY'
import socket
import sys
with socket.socket() as sock:
    sock.bind((sys.argv[1], int(sys.argv[2])))
PY
if [[ -f "$PIPER_LOG/server.log" ]]; then
    cp "$PIPER_LOG/server.log" "$PIPER_LOG/server.$(date +%Y%m%d_%H%M%S).log"
fi
# 子进程不继承启动锁；模型就绪与否由检查脚本的 ping 确认。
nohup "$PIPER_ENV/bin/python" -u scripts/serve_piper_x.py \
    --ckpt-dir "$PIPER_MODEL" --ckpt-name "$PIPER_CKPT" \
    --host "$PIPER_HOST" --port "$PIPER_PORT" --device "$PIPER_DEVICE" \
    --compile-enabled false --denoise-steps 10 --inference-mode sync \
    inference.action_response_mode=chunk \
    >"$PIPER_LOG/server.log" 2>&1 </dev/null 9>&- &
echo $! >"$PIPER_LOG/server.pid"
echo "已发起 Piper-X 加载：PID=$(cat "$PIPER_LOG/server.pid")"
echo "日志：$PIPER_LOG/server.log"

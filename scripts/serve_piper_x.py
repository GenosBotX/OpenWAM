"""启动 Piper-X WebSocket 服务，命令行参数复用 OpenWAM 原生入口。"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "third_party")]

from openwam.deploy.piper_x import PiperXPolicyServer
from openwam.deploy.server import main


if __name__ == "__main__":
    main(server_class=PiperXPolicyServer)

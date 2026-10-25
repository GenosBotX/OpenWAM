"""加载 TOS 凭证并执行命令，密钥不会写入仓库或命令行。

优先使用环境中完整的 TOS_ACCESS_KEY / TOS_SECRET_KEY；否则读取外部
JSON 文件。文件还可包含 TOS_SECURITY_TOKEN、TOS_ENDPOINT、TOS_REGION。
用法：python scripts/run_with_tos_credentials.py /secure/tos.json python ...
"""

import json
import os
from pathlib import Path
import stat
import sys


class CredentialError(ValueError):
    pass


def load_credentials(path):
    required = {"TOS_ACCESS_KEY", "TOS_SECRET_KEY"}
    if any(key in os.environ for key in required):
        if not all(os.environ.get(key, "").strip() for key in required):
            raise CredentialError(
                "请在训练任务环境中同时设置非空的 TOS_ACCESS_KEY 和 TOS_SECRET_KEY；"
                "不会与文件中的另一套凭证混用。"
            )
        return

    path = Path(path).expanduser()
    with path.open() as source:
        info = os.fstat(source.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077):
            raise CredentialError("凭证必须为当前用户拥有的普通文件，权限请设为 0600。")
        try:
            values = json.load(source)
        except (ValueError, UnicodeError):
            raise CredentialError("凭证文件必须为有效的 UTF-8 JSON 对象。") from None

    allowed = required | {"TOS_SECURITY_TOKEN", "TOS_ENDPOINT", "TOS_REGION"}
    if (not isinstance(values, dict) or not required.issubset(values)
            or not set(values).issubset(allowed)):
        raise CredentialError(
            "凭证 JSON 必须包含 TOS_ACCESS_KEY 和 TOS_SECRET_KEY；"
            "仅允许附加 TOS_SECURITY_TOKEN、TOS_ENDPOINT、TOS_REGION。"
        )
    if not all(isinstance(value, str) and value.strip() for value in values.values()):
        raise CredentialError("凭证 JSON 中的值必须是非空字符串。")

    # AK、SK、临时 token 必须来自同一来源，避免残留 token 污染文件凭证。
    for key in required | {"TOS_SECURITY_TOKEN"}:
        os.environ.pop(key, None)
        if key in values:
            os.environ[key] = values[key]
    # endpoint / region 是连接配置，允许由任务环境覆盖文件中的默认值。
    for key in ("TOS_ENDPOINT", "TOS_REGION"):
        if key in values and not os.environ.get(key):
            os.environ[key] = values[key]


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    try:
        load_credentials(sys.argv[1])
    except FileNotFoundError:
        raise SystemExit(
            f"TOS 凭证文件不存在：{sys.argv[1]}。请通过 OPENWAM_TOS_CREDENTIALS "
            "或 TOS_CREDENTIAL_FILE 指定 worker 可读的共享文件，"
            "或在训练任务环境中注入 TOS_ACCESS_KEY 和 TOS_SECRET_KEY。"
            "SSH 登录容器的 /root 文件不会自动同步到训练 worker。"
        ) from None
    except CredentialError as exc:
        raise SystemExit(f"TOS 凭证加载失败：{exc}") from None
    except OSError as exc:
        raise SystemExit(
            f"无法读取 TOS 凭证文件（errno={exc.errno}），请检查 worker 的挂载和访问权限。"
        ) from None
    os.execvp(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    main()

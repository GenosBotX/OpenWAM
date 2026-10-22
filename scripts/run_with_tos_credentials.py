"""Read an external mode-0600 JSON credential file, then exec a command.

JSON keys: TOS_ACCESS_KEY, TOS_SECRET_KEY, optionally TOS_SECURITY_TOKEN,
TOS_ENDPOINT and TOS_REGION. No credentials are stored in this repository.
Usage: python scripts/run_with_tos_credentials.py /secure/tos.json python ...
"""

import json
import os
from pathlib import Path
import stat
import sys


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1])
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("Credential file must be owned by current user and mode 0600")
    allowed = {"TOS_ACCESS_KEY", "TOS_SECRET_KEY", "TOS_SECURITY_TOKEN", "TOS_ENDPOINT", "TOS_REGION"}
    values = json.loads(path.read_text())
    if not {"TOS_ACCESS_KEY", "TOS_SECRET_KEY"}.issubset(values) or not set(values).issubset(allowed):
        raise ValueError("Unexpected credential JSON keys")
    os.environ.update({k: str(v) for k, v in values.items()})
    os.execvp(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    main()

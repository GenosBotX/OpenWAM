#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Height x width of the assembled multiview frame.
export LINGLONG_TOS_OUTPUT_PATH="${LINGLONG_TOS_OUTPUT_PATH:-/vePFS/tangyuting/OpenWAM/outputs/linglong_tos_0923_1017_eef_8gpu_bs6_acc24_h288_w256}"
echo "Video input: height=288, width=256 (assembled multiview frame)"
exec bash "$repo_root/scripts/launch_volcano_linglong_tos_eef_8gpu_bs6.sh" \
  dataloader.height=288 dataloader.width=256 "$@"

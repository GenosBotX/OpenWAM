#!/usr/bin/env bash
set -euo pipefail

# 8 GPUs x batch 4 x accumulation 24 = effective batch 768. Does not resume.
# Uses the all-stuff prompt copy and a separate output directory.
export NPROC=8
export EXPECTED_WORLD_SIZE=8
export NNODES=1
export CONFIG_NAME=train_linglong_0923_1010_1012_1017_fittings_eef_gtstate_allstuff_8gpu
export DATASET_DIR=/iag_ad_01/ad/xiejinxia/code/test/OpenWAM/data/linglong_0923_1010_1012_1017_fittings_eef_gtstate_allstuff_train_15hz
for artifact in manifest.json meta/normalization_stats.npy meta/state_eef_stats.json meta/action_eef_stats.json; do
  if [[ ! -f "$DATASET_DIR/$artifact" ]]; then
    echo "ERROR: all-stuff EEF bucket is not ready: $DATASET_DIR/$artifact" >&2
    exit 2
  fi
done
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/launch_volcano_linglong_0923_1010_32gpu.sh" "$@"

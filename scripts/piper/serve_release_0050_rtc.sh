#!/usr/bin/env bash
# Standalone equivalent of the model-panel release 0050 settings.
set -euo pipefail
rtc_release_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$rtc_release_root"
export OPENWAM_REPO_ROOT="$rtc_release_root"
export OPENWAM_BASE_CHECKPOINT="${OPENWAM_BASE_CHECKPOINT:-/home/zhuoyu/piper-x/OpenWAM-Alpha-Real-RoboDojo-Piper-X/checkpoint_step_30000.safetensors}"
exec "${RTC_MODEL_PYTHON:-/home/zhuoyu/tangyuting/envs/openwam-fp8/bin/python}" -u scripts/deploy.py \
  --config configs/deploy_release_0050_rtc_fp8_5step.yaml \
  --ckpt-dir "${RTC_CHECKPOINT_DIR:-/home/zhuoyu/liqi/ckpt/1111_rtc_p16_based_0030}" \
  --ckpt-name checkpoint_step_18495.safetensors \
  --device cuda:0 --host 0.0.0.0 --port 8848 \
  inference.height=384 inference.width=320 \
  optimization.dit_cache.mode=fixed \
  'optimization.dit_cache.recompute_steps=[1,3]' \
  "$@"

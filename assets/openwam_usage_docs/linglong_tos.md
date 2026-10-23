# Linglong Training From TOS

This repo keeps the Linglong EEF training contract and reads immutable converted
Linglong buckets from TOS. Parquet, normalization metadata, and JPEGs are fetched
on demand into a per-node disk cache shared by all local ranks and DataLoader
workers. The default production quota is 4 GiB per node, with a 1 GiB free-space
guard. Cache settings must match for every process sharing a cache directory.

## Publish A Dataset Version

Install the optional client and provide credentials through environment variables
or a mode-0600 JSON credential file outside the repo:

```bash
pip install -e '.[tos]'
export TOS_ACCESS_KEY=...
export TOS_SECRET_KEY=...
```

Publish a small fixture first, then a full version under a new prefix. The script
uploads referenced images and parquet files, then metadata, and writes
`manifest.json` last. A prefix with a manifest is immutable and cannot be reused.
The subset keeps the original full training EEF normalization statistics.

```bash
python -m scripts.publish_linglong_tos \
  --source /vePFS/data/linglong/CONVERTED_TRAIN_BUCKET \
  --destination tos://BUCKET/linglong/openwam/linglong/2026-10-v1 \
  --episodes 2

# Full dataset: omit --episodes. Upload progress includes bytes and object count.
python -m scripts.publish_linglong_tos \
  --source /vePFS/data/linglong/CONVERTED_TRAIN_BUCKET \
  --destination tos://BUCKET/linglong/openwam/linglong/2026-10-v2
```

Set `TOS_ENDPOINT` and `TOS_REGION` only when the bucket is outside the default
Beijing region. `TOS_SECURITY_TOKEN` is supported for temporary credentials.

## Start Training

The full recipe uses the 16D `observation.state_eef` / `action_eef` pair, the
existing EEF normalization artifacts, 33 control frames, video stride 4, and
window stride 1 (matching the local Linglong Base recipe). The default cache is shared at `/tmp/openwam-tos-cache` on each
node. Override `OPENWAM_TOS_CACHE_DIR` and the cache quota in the config for the
node's local disk budget.

```bash
export LINGLONG_TOS_DATASET=tos://BUCKET/linglong/openwam/linglong/2026-10-v2
export TOS_ACCESS_KEY=...
export TOS_SECRET_KEY=...
NPROC_PER_NODE=8 scripts/train_linglong_tos.sh
```

On Volcano, expose credentials through the job's secret environment or point
`TOS_CREDENTIAL_FILE` at a JSON file owned by the job user with mode `0600`.
The launcher validates the dataset URI and uses the regular OpenWAM training
entrypoint. Checkpoints are written to `training.output_path` on local/vePFS
storage. Upload completed checkpoints separately after the run; active checkpoint
directories are not mirrored to TOS.

For a bounded end-to-end smoke test, set `OPENWAM_TOS_CONFIG=train_linglong_tos_smoke`.
It runs one optimizer step with batch size 1 and an 8 MiB cache to exercise
eviction. It freezes the video backbone and trains ActionDiT plus the proprio
encoder on two 80 GB GPUs, disables AdamW foreach temporary buffers, and writes
no training checkpoint. This validates the data/training path; use the full
recipe on the usual larger GPU allocation for joint video/action fine-tuning.

## Cache Behavior

- Objects are keyed by their full TOS URI and downloaded with ETag matching.
- Per-object striped file locks deduplicate concurrent misses and pin active reads.
- A SQLite LRU index reserves space before download and evicts only unleased files.
- Oversized objects and insufficient free space fail explicitly; the reader never
  silently falls back to a local dataset copy.
- The TOS dataset prefix must remain immutable. Do not overwrite objects after
  publishing `manifest.json`.
- Cache statistics and the last training step's loss/gradient are written to
  `tos_training_report.json` in the actual run directory (the trainer creates a
  timestamped child under `training.output_path`). The quota covers cached object
  payloads and in-flight reservations; SQLite and lock files add small overhead.

`python -m scripts.check_linglong_tos` compares actions, state, masks, prompts,
and every rendered multiview frame against the original vePFS dataset. The full
dataset is not copied for this check.

## Validation On 2026-10-22

The isolated published fixture contains the first two complete episodes: 1,130
objects, approximately 52.2 MB. Full data has **not** been published.

Dataset URI:
`tos://ml-platform-auto-created-required-2109640594-cn-beijing/linglong/openwam/linglong/20261022_smoke_v1`

- Eight windows exactly matched the original vePFS data: EEF action/state,
  masks, prompts and all decoded multiview frames. With a 64 MiB cache, first
  pass took 12.882 s, repeat pass 2.728 s, and two-worker pass 0.881 s.
- Two A800 80 GB GPUs completed one real optimizer step using the Foundation
  checkpoint, bf16, ZeRO-2, batch 1/rank, accumulation 1, and frozen video backbone.
  The step took approximately 6.1 s, excluding several minutes of model loading
  and setup. Total loss was 0.3003464 (video 0.2878386, action 0.0125078), with
  gradient norm 0.2334837. This is a smoke result, not a throughput estimate for
  full training. GPU peak memory was not recorded for this successful run.
- The 8 MiB training cache downloaded 9,250,049 bytes, recorded 137 hits / 204
  misses / 24 evictions, and ended at 8,384,739 payload bytes, below its 8,388,608
  byte limit. DataLoader prefetch contributes to these totals.
- Full video/action fine-tuning on two GPUs reached backward but failed in
  AdamW's optimizer step (an additional 11.22 GiB allocation with 9.97 GiB free),
  even with `optimizer_foreach=false`. CPU offload was also blocked by the
  node's CUDA toolkit/PyTorch build mismatch. Neither full two-GPU training nor
  larger-scale throughput is validated by this smoke run.

Remote validation logs and JSON reports are under
`/vePFS/tangyuting/OpenWAM-tos/outputs/tos_validation/`. Training checkpoints
remain local/vePFS outputs; automatic checkpoint upload to TOS is not implemented.

Reproduce the successful smoke test from the remote repo:

```bash
cd /vePFS/tangyuting/OpenWAM-tos
export TOS_CREDENTIAL_FILE=/root/.config/openwam/tos-linglong.json
export LINGLONG_TOS_DATASET=tos://ml-platform-auto-created-required-2109640594-cn-beijing/linglong/openwam/linglong/20261022_smoke_v1
export OPENWAM_TOS_CONFIG=train_linglong_tos_smoke
NPROC_PER_NODE=2 scripts/train_linglong_tos.sh
```

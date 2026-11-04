"""Validate Piper-X entry files through the same bounded storage as training."""
import json
import os

from openwam.dataloader.tos_cache import BoundedTosCache, DatasetStorage, is_remote_path

VERSIONS = (
    "20261022_eef_v1", "20261023_eef_v1", "20261024_eef_v1",
    "20261026_eef_v2", "20261026_static_eef_v2",
    "20261027_eef_v1", "20261027_static_eef_v1",
)


def main():
    root = os.environ.get("OPENWAM_DATA_ROOT", "obs://genosbot-shanghai/openwam/data/piper-x").rstrip("/")
    cache = BoundedTosCache(
        os.environ.get("OPENWAM_CACHE_DIR", "/tmp/openwam-piper-obs-cache"),
        int(os.environ.get("OPENWAM_CACHE_MAX_BYTES", 4 * 1024**3)),
        int(os.environ.get("OPENWAM_CACHE_MIN_FREE_BYTES", 1024**3)),
    ) if is_remote_path(root) else None
    for version in VERSIONS:
        for split in ("train", "test"):
            storage = DatasetStorage(f"{root}/{version}/{split}", cache)
            manifest = json.loads(storage.text("manifest.json"))
            if manifest.get("status") != "complete" or manifest.get("split") != split:
                raise ValueError(f"Incomplete or wrong split: {version}/{split}")
            episodes = [json.loads(row) for row in storage.text("meta/episodes.jsonl").splitlines() if row.strip()]
            if not episodes:
                raise ValueError(f"Empty episodes: {version}/{split}")
            print(f"DATA_OK {version}/{split}: {len(episodes)} episodes", flush=True)
    storage = DatasetStorage(f"{root}/20261022_20261027_normalization_v1", cache)
    stats = json.loads(storage.text("meta/eef20_normalization_stats.json"))
    if stats.get("format") != "piper_x_eef20_normalization_v1" or not stats.get("test_excluded"):
        raise ValueError("Expected training-only EEF20 normalization")
    print("NORMALIZATION_OK", flush=True)
    if cache:
        print("CACHE", json.dumps(cache.stats()), flush=True)


if __name__ == "__main__":
    main()

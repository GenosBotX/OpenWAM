"""Publish converted portable Linglong buckets to a NEW immutable TOS prefix.

Run with python -m scripts.publish_linglong_tos --help. Credentials come from
TOS_ACCESS_KEY/TOS_SECRET_KEY, optionally TOS_SECURITY_TOKEN.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import pyarrow.parquet as pq

from openwam.dataloader.tos_cache import TosBackend, join_tos, parse_tos_uri


def publish(source, destination, episodes=None, workers=8):
    source = Path(source).resolve()
    bucket, prefix = parse_tos_uri(destination)
    if not prefix:
        raise ValueError("Use a dedicated versioned dataset prefix")
    backend = TosBackend()
    # Refuse to modify a published version. Failed/incomplete uploads can resume.
    listing = backend.client.list_objects_type2(bucket, prefix=prefix + "/manifest.json", max_keys=1)
    if any(x.key == prefix + "/manifest.json" for x in listing.contents):
        raise ValueError("Destination is already published; choose a NEW immutable prefix")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Source dataset is incomplete")
    rows = [json.loads(x) for x in (source / "meta/episodes.jsonl").read_text().splitlines() if x.strip()]
    if episodes is not None:
        if episodes < 1:
            raise ValueError("episodes must be positive")
        rows = rows[:episodes]
    image_root = Path(manifest.get("source_root", "."))
    if not image_root.is_absolute():
        image_root = source / image_root
    total_files, total_bytes = 0, 0
    started = time.monotonic()

    def upload(item):
        rel, path = item
        # Validate object keys and prevent escaping through source image paths.
        uri = join_tos(destination, rel)
        b, key = parse_tos_uri(uri)
        backend.client.put_object_from_file(b, key, str(path))
        return path.stat().st_size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Bound memory to one episode's image list; do not walk millions of files.
        for local, row in enumerate(rows):
            if row["episode_index"] != local:
                raise ValueError("Expected contiguous episode indices starting at zero")
            rel = f"data/chunk-{local // 1000:03d}/episode_{local:06d}.parquet"
            table = pq.read_table(source / rel, columns=[f"observation.images.{cam}" for cam in ("head", "left", "right")])
            if table.num_rows != row["length"]:
                raise ValueError("Episode metadata/parquet row count mismatch")
            images = set()
            for column in table.columns:
                images.update(column.to_pylist())
            files = [(rel, source / rel)] + [(x, image_root / x) for x in sorted(images)]
            for size in pool.map(upload, files):
                total_files += 1
                total_bytes += size
            print(json.dumps({"episode": local, "files": total_files, "bytes": total_bytes}), flush=True)
        metadata = [(str(p.relative_to(source)), p) for p in sorted((source / "meta").iterdir()) if p.is_file()]
        # Episode-specific metadata must follow the selected subset.
        metadata = [(rel, p) for rel, p in metadata if rel not in ("meta/episodes.jsonl", "meta/episodes_stats.jsonl")]
        for size in pool.map(upload, metadata):
            total_files += 1
            total_bytes += size
    def put_text(relative, content):
        backend.client.put_object(bucket, prefix + "/" + relative, content=content.encode())
    put_text("meta/episodes.jsonl", "".join(json.dumps(x) + "\n" for x in rows))
    stats_path = source / "meta/episodes_stats.jsonl"
    if stats_path.is_file():
        put_text("meta/episodes_stats.jsonl", "\n".join(stats_path.read_text().splitlines()[:len(rows)]) + "\n")
    manifest.update(source_root=".", total_episodes=len(rows), total_frames=sum(x["length"] for x in rows),
                    total_windows=sum(max(0, x["length"] - 32) for x in rows),
                    normalization_dataset_dir=".", tos_immutable=True)
    manifest["tos_publication"] = {"files_before_manifest": total_files, "bytes_before_manifest": total_bytes,
                                   "subset": episodes is not None, "normalization": "original full training bucket statistics"}
    # Commit marker is written only after every referenced object succeeds.
    put_text("manifest.json", json.dumps(manifest, indent=2) + "\n")
    result = {"destination": destination, "episodes": len(rows), "files_before_manifest": total_files,
              "bytes_before_manifest": total_bytes, "seconds": round(time.monotonic() - started, 3)}
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--episodes", type=int, help="First N complete episodes for a smoke test; omit for full publication")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    publish(args.source, args.destination, args.episodes, args.workers)

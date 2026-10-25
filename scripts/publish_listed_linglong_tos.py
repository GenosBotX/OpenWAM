"""Publish the listed Linglong directories, verify them, then delete local copies.

Training buckets use the official publisher. The 2026-10-19 decoded tree has no
OpenWAM manifest, so it is stored as a directory tree under its own prefix.
Deletion runs only after every prefix passes the size and count checks.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openwam.dataloader.tos_cache import BoundedTosCache, DatasetStorage, TosBackend, join_tos

BASE = "tos://ml-platform-auto-created-required-2109640594-cn-beijing/linglong/openwam/linglong"
ROOT = Path("/vePFS/data/linglong")
PYTHON = "/vePFS/tangyuting/envs/openwam/bin/python"
WORKERS = 24
PARALLEL = 2

JOBS = [
    ("bucket", ROOT / "linglong_1023_openwam/linglong_openwam_train_15hz", f"{BASE}/20261024_1023_train_v1"),
    ("bucket", ROOT / "linglong_1023_openwam/linglong_openwam_test_15hz", f"{BASE}/20261024_1023_test_v1"),
    ("bucket", ROOT / "linglong_20261020_openwam/linglong_openwam_train_15hz", f"{BASE}/20261024_20261020_train_v1"),
    ("bucket", ROOT / "linglong_20261020_openwam/linglong_openwam_test_15hz", f"{BASE}/20261024_20261020_test_v1"),
    ("bucket", ROOT / "linglong_20261019_20261020_openwam/linglong_openwam_train_15hz", f"{BASE}/20261024_sheet_20261019_20261020_train_v1"),
    ("bucket", ROOT / "linglong_20261019_20261020_openwam/linglong_openwam_test_15hz", f"{BASE}/20261024_sheet_20261019_20261020_test_v1"),
    ("bucket", ROOT / "linglong_20261019_20261020_20261023_openwam/linglong_openwam_train_15hz", f"{BASE}/20261024_sheet_20261019_20261020_20261023_train_v1"),
    ("bucket", ROOT / "linglong_20261019_20261020_20261023_openwam/linglong_openwam_test_15hz", f"{BASE}/20261024_sheet_20261019_20261020_20261023_test_v1"),
    ("bucket", ROOT / "linglong_0923_1010_1012_1017_fittings_eef_gtstate_portable/data/linglong_0923_1010_1012_1017_fittings_eef_gtstate_train_15hz", f"{BASE}/20261024_fittings_portable_train_v1"),
    ("bucket", ROOT / "linglong_0923_1010_1012_1017_fittings_eef_gtstate_portable/data/linglong_0923_1010_1012_1017_fittings_eef_gtstate_test_15hz", f"{BASE}/20261024_fittings_portable_test_v1"),
    ("bucket", ROOT / "linglong_0923_1010_1012_1017_fittings_eef_gtstate_portable/data/linglong_0923_1010_1012_1017_fittings_eef_gtstate_allstuff_train_15hz", f"{BASE}/20261024_fittings_portable_allstuff_train_v1"),
    ("bucket", ROOT / "linglong_0923_1010_1012_1017_fittings_20261019_eef_gtstate_train_15hz", f"{BASE}/20261024_fittings_20261019_train_v1"),
    ("bucket", ROOT / "linglong_allstuff_20261019_20261020_eef_train_15hz", f"{BASE}/20261024_allstuff_20261019_20261020_train_v1"),
    ("tree", ROOT / "linglong_20261019_processed", f"{BASE}/20261024_20261019_processed_v1"),
]

DELETE_DIRS = [
    ROOT / "linglong_0923_1010_1012_1017_fittings_20261019_eef_gtstate_train_15hz",
    ROOT / "linglong_0923_1010_1012_1017_fittings_eef_gtstate_portable",
    ROOT / "linglong_1023_openwam",
    ROOT / "linglong_20261019_20261020_20261023_openwam",
    ROOT / "linglong_20261019_20261020_openwam",
    ROOT / "linglong_20261019_processed",
    ROOT / "linglong_20261020_openwam",
    ROOT / "linglong_allstuff_20261019_20261020_eef_train_15hz",
]
DELETE_ARCHIVES = [
    ROOT / "linglong_0923_1010_1012_1017_fittings_eef_gtstate_portable.tar.zst",
    ROOT / "raw" / "linglong_1023.tar.gz",
    ROOT / "linglong_1023.tar.gz",
    ROOT / "linglong_2026-10-19.tar.gz",
    ROOT / "sampler_data_2026-10-20.tar",
]


def _marker(kind: str) -> str:
    return "manifest.json" if kind == "bucket" else "publication.json"


def _published(destination: str, kind: str) -> bool:
    from openwam.dataloader.tos_cache import parse_tos_uri

    bucket, prefix = parse_tos_uri(destination)
    key = prefix + "/" + _marker(kind)
    page = TosBackend().client.list_objects_type2(bucket, prefix=key, max_keys=1)
    return any(item.key == key for item in page.contents)


def _publish(kind: str, source: Path, destination: str) -> None:
    if _published(destination, kind):
        print(json.dumps({"skip_existing": destination}), flush=True)
        return
    command = [
        PYTHON, "-m", "scripts.publish_linglong_tos",
        "--source", str(source),
        "--destination", destination,
        "--workers", str(WORKERS),
        "--resume",
    ]
    if kind == "tree":
        command.append("--tree")
    print(json.dumps({"start": destination, "source": str(source)}), flush=True)
    subprocess.check_call(command, cwd="/vePFS/zhuoyu/OpenWAM")


def _local_image(bucket: Path):
    import pyarrow.parquet as pq

    manifest = json.loads((bucket / "manifest.json").read_text())
    table = pq.read_table(bucket / "data/chunk-000/episode_000000.parquet", columns=["observation.images.head"])
    relative = table["observation.images.head"][0].as_py()
    image_root = Path(manifest.get("source_root", "."))
    if not image_root.is_absolute():
        image_root = bucket / image_root
    return relative, image_root / relative


def _verify_bucket(source: Path, destination: str) -> dict:
    local = json.loads((source / "manifest.json").read_text())
    cache = BoundedTosCache("/tmp/openwam-tos-publish-verify", 512 * 1024**2, 256 * 1024**2)
    remote = json.loads(DatasetStorage(destination, cache).text("manifest.json"))
    checks = {
        "status": remote.get("status") == "complete",
        "split": remote.get("split") == local.get("split"),
        "episodes": remote.get("total_episodes") == local.get("total_episodes"),
        "frames": remote.get("total_frames") == local.get("total_frames"),
        "source_root": remote.get("source_root") == ".",
    }
    backend = TosBackend()
    parquet = "data/chunk-000/episode_000000.parquet"
    remote_size, _ = backend.head(join_tos(destination, parquet))
    checks["parquet_bytes"] = remote_size == (source / parquet).stat().st_size
    image_rel, image_path = _local_image(source)
    image_size, _ = backend.head(join_tos(destination, image_rel))
    checks["image_bytes"] = image_size == image_path.stat().st_size and image_size > 0
    rows = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    last = len(rows) - 1
    last_rel = f"data/chunk-{last // 1000:03d}/episode_{last:06d}.parquet"
    last_size, _ = backend.head(join_tos(destination, last_rel))
    checks["last_parquet_bytes"] = last_size == (source / last_rel).stat().st_size
    return {"destination": destination, "ok": all(checks.values()), "checks": checks}


def _verify_tree(source: Path, destination: str) -> dict:
    cache = BoundedTosCache("/tmp/openwam-tos-publish-verify", 512 * 1024**2, 256 * 1024**2)
    remote = json.loads(DatasetStorage(destination, cache).text("publication.json"))
    local_files = [path for path in source.rglob("*") if path.is_file()]
    local_bytes = sum(path.stat().st_size for path in local_files)
    checks = {
        "status": remote.get("status") == "complete",
        "files": remote.get("files") == len(local_files),
        "bytes": remote.get("bytes") == local_bytes,
    }
    sample = local_files[0]
    remote_size, _ = TosBackend().head(join_tos(destination, sample.relative_to(source).as_posix()))
    checks["sample_bytes"] = remote_size == sample.stat().st_size and remote_size > 0
    return {"destination": destination, "ok": all(checks.values()), "checks": checks}


def _entropy_still_reading() -> bool:
    """The 1023 entropy worker reads the local 1023 bucket until it leaves the queue."""
    command = ["/root/.volc/bin/volc", "ml_task", "get", "-i", "t-20261024143909-cbf8q", "-o", "json"]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"entropy_status": "unknown", "error": str(exc)[:300]}), flush=True)
        return True
    start = output.find("[{")
    if start < 0:
        start = output.rfind("{")
    payload = json.loads(output[start:])
    row = payload[0] if isinstance(payload, list) else payload
    status = row.get("Status")
    print(json.dumps({"entropy_task": "t-20261024143909-cbf8q", "status": status}), flush=True)
    return status in {"Queue", "Staging", "Running", "FailedHolding"}


def main() -> None:
    missing = [str(source) for _, source, _ in JOBS if not source.exists()]
    if missing:
        raise SystemExit(f"missing sources: {missing}")
    record_dir = Path("/vePFS/zhuoyu/OpenWAM/logs/tos_publish")
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "destinations.json").write_text(json.dumps(
        [{"kind": kind, "source": str(source), "destination": destination} for kind, source, destination in JOBS],
        indent=2,
    ) + "\n")
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = {pool.submit(_publish, kind, source, destination): destination for kind, source, destination in JOBS}
        for future in as_completed(futures):
            future.result()
    reports = []
    for kind, source, destination in JOBS:
        report = _verify_bucket(source, destination) if kind == "bucket" else _verify_tree(source, destination)
        reports.append(report)
        print(json.dumps(report), flush=True)
    record = Path("/vePFS/zhuoyu/OpenWAM/logs/tos_publish/verification.json")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(reports, indent=2) + "\n")
    if not all(item["ok"] for item in reports):
        raise SystemExit("verification failed; local data was kept")
    hold_1023 = _entropy_still_reading()
    for path in DELETE_DIRS:
        if hold_1023 and path.name == "linglong_1023_openwam":
            print(json.dumps({"kept_until_entropy_finishes": str(path)}), flush=True)
            continue
        if path.exists():
            shutil.rmtree(path)
            print(json.dumps({"deleted_dir": str(path)}), flush=True)
    for path in DELETE_ARCHIVES:
        if path.is_file():
            path.unlink()
            print(json.dumps({"deleted_archive": str(path)}), flush=True)
    print(json.dumps({"status": "published_verified_deleted", "prefixes": [job[2] for job in JOBS]}), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, "/vePFS/zhuoyu/OpenWAM")
    main()

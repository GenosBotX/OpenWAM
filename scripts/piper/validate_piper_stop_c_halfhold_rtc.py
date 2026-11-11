"""Validate halfhold data and RTC configuration without loading a model/GPU."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from openwam.dataloader.piper_x import PiperXDataset
from openwam.model.rtc import validate_prefix_lengths

CONFIG = "train_piper_stop_c_halfhold_relative_eef_rtc_p16_16gpu_e5"
EXPECTED = dict(dynamic=259633, transition=53258, original_static=24854, synthetic_hold=17309)


def validate(config, decode_samples=False):
    cfg = OmegaConf.load(config)
    lengths = validate_prefix_lengths(cfg.training.rtc.prefix_lengths, 32)
    if not cfg.training.rtc.enabled or list(lengths) != [0, *range(6, 17)]:
        raise ValueError("Expected RTC prefix lengths 0 and 6..16")
    if (cfg.training.batch_size, cfg.training.gradient_accumulation_steps, cfg.training.num_epochs) != (6, 8, 5):
        raise ValueError("Expected batch=6, accumulation=8, epochs=5")
    if cfg.training.resume_ckpt_path is not None or not str(cfg.training.finetune_ckpt_path).endswith("checkpoint_step_30000.safetensors"):
        raise ValueError("Halfhold RTC must start from the original Piper-X base")
    dataset = PiperXDataset.from_config(cfg.dataloader)
    counts = Counter()
    sources = []
    episodes = 0
    for ds in dataset.datasets:
        manifest = ds._read_manifest(ds._storage, "train")
        if manifest.get("stop_ablation_version") != "C_halfhold_v1" or not manifest.get("window_selection"):
            raise ValueError("Expected published C_halfhold_v1 selected windows")
        entries = json.loads(ds._storage.text(manifest["window_selection"]))["entries"]
        indices = {}
        cursor = 0
        for entry in entries:
            counts[entry["kind"]] += entry["repeats"]
            indices.setdefault(entry["kind"], cursor)
            cursor += entry["repeats"]
        if cursor != len(ds):
            raise ValueError("Dataset reader lost selected-window weights")
        if decode_samples:
            for index in indices.values():
                sample = ds[index]
                if sample["action"].shape != (32, 80) or len(sample["video"]) != 9:
                    raise ValueError("Invalid decoded action/video shape")
                if not np.isfinite(sample["action"].numpy()).all():
                    raise ValueError("Nonfinite decoded action")
        episodes += len(ds._episodes)
        sources.append(dict(uri=str(ds.dataset_dir), images=str(ds._source_root), windows=len(ds),
                            decoded_kinds=list(indices) if decode_samples else []))
    if len(dataset.datasets) != 10 or len(dataset) != 355054 or episodes != 686 or dict(counts) != EXPECTED:
        raise ValueError(f"Halfhold sampling changed: {len(dataset)}, {episodes}, {dict(counts)}")
    for item in cfg.dataloader.eval_dataset_dir:
        if "/20261105_stop_C_halfhold_v1/" in item or not item.endswith("/test"):
            raise ValueError("Evaluation must retain original test buckets")
    microsteps = math.ceil(math.ceil(len(dataset) / 16) / cfg.training.batch_size)
    return dict(status="passed", samples=len(dataset), episodes=episodes, counts=dict(counts),
                prefix_lengths=list(lengths), nodes=2, gpus=16, microsteps_per_epoch=microsteps,
                total_microsteps=microsteps * cfg.training.num_epochs, sources=sources)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / (CONFIG + ".yaml"))
    parser.add_argument("--decode-samples", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.config, args.decode_samples)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")

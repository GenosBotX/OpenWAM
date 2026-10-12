"""Recompute body-part MAE from saved, denormalized Linglong predictions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PARTS = {
    "左臂": list(range(0, 7)),
    "左夹爪": [7],
    "右臂": list(range(8, 15)),
    "右夹爪": [15],
    "腰部": list(range(16, 20)),
    "头部": [20, 21],
}


def summarize(error):
    per_dim = np.abs(error).mean(axis=(0, 1))
    return {
        "windows": int(error.shape[0]),
        "action_timesteps": int(error.shape[0] * error.shape[1]),
        "per_dim_mae": per_dim.tolist(),
        "body_parts": {
            name: {"raw_indices": dims, "num_dims": len(dims),
                   "mae": float(per_dim[dims].mean())}
            for name, dims in PARTS.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir", type=Path)
    args = parser.parse_args()
    root = args.eval_dir
    provenance = json.loads((root / "aggregate_metrics.json").read_text())
    errors, episodes = [], {}
    for path in sorted(root.glob("episode_*/metrics.npz")):
        with np.load(path, allow_pickle=False) as data:
            pred = data["predictions"].astype(np.float64)
            target = data["targets"].astype(np.float64)
            starts = data["starts"]
        if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 24:
            raise ValueError(f"Unexpected array shape: {path}: {pred.shape}, {target.shape}")
        if len(starts) != len(pred) or len(np.unique(starts)) != len(starts):
            raise ValueError(f"Invalid or repeated windows: {path}")
        error = pred - target
        if not np.isfinite(error).all():
            raise ValueError(f"Non-finite error: {path}")
        errors.append(error)
        episodes[path.parent.name] = summarize(error)
    if len(episodes) != provenance["episodes"]:
        raise ValueError("Episode count does not match original evaluation")
    joined = np.concatenate(errors)
    if len(joined) != provenance["windows"]:
        raise ValueError("Window count does not match original evaluation")
    aggregate = summarize(joined)
    np.testing.assert_allclose(aggregate["per_dim_mae"][:22],
                               provenance["per_dim_mae"], atol=1e-6, rtol=1e-5)
    result = {
        "source_eval_dir": str(root.resolve()),
        "checkpoint_dir": provenance["checkpoint_dir"],
        "dataset_dir": provenance["dataset_dir"],
        "method": "MAE per raw dimension over all evaluated action timesteps, then arithmetic mean across dimensions within each body part",
        "weighting": "All action timesteps equally weighted; episodes are not equally weighted",
        "units": "Original denormalized action units; no unit conversion",
        "excluded_raw_indices": [22, 23],
        "inference_rerun": False,
        "episodes": len(episodes),
        "aggregate": aggregate,
        "per_episode": episodes,
    }
    out = root / "body_part_metrics"
    out.mkdir(exist_ok=True)
    (out / "body_part_mae.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "body_part_mae.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "part", "num_dims", "raw_indices", "mae", "windows", "action_timesteps"])
        for scope, metrics in [("all_test", aggregate), *episodes.items()]:
            for name, part in metrics["body_parts"].items():
                writer.writerow([scope, name, part["num_dims"], str(part["raw_indices"]),
                                 part["mae"], metrics["windows"], metrics["action_timesteps"]])
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

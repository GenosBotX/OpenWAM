"""Evaluate one OpenWAM checkpoint on every Linglong test episode.

The model is loaded once, while each test episode is evaluated with
non-overlapping action chunks (normally stride 32).  Per-episode metrics and
optional GT/generated MP4s are written below ``--output-dir``.
"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--chunk-stride", type=int, default=32)
    p.add_argument("--max-windows", type=int, default=1000)
    p.add_argument("--denoise-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--decode-video", action="store_true")
    args = p.parse_args()
    if args.chunk_stride <= 0 or args.max_windows <= 0:
        p.error("chunk-stride and max-windows must be positive")

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from openwam.dataloader.linglong import LinglongDataset
    from openwam.deploy.engine import JointInferenceEngine
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.deploy.server import _load_deploy_yaml, merge_deploy_cfg

    print(f"[eval] loading checkpoint {args.checkpoint_dir}", flush=True)
    cfg, architecture = load_from_checkpoint_dir(args.checkpoint_dir, device="cuda:0")
    deploy_cfg = _load_deploy_yaml()
    deploy_cfg.inference.denoise_steps = int(args.denoise_steps)
    deploy_cfg.optimization.compile.enabled = False
    deploy_cfg.optimization.decode_video = bool(args.decode_video)
    cfg = merge_deploy_cfg(cfg, deploy_cfg)
    engine = JointInferenceEngine(cfg=cfg, architecture=architecture)

    dl_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataloader, resolve=True))
    dl_cfg.dataset_dir = args.dataset_dir
    dl_cfg.overfit_single = False
    dl_cfg.overfit_repeats = 1
    dl_cfg.window_stride = args.chunk_stride
    ds = LinglongDataset.from_config(dl_cfg, split="test")
    if ds.dataset_dir.resolve() != Path(args.dataset_dir).resolve():
        raise ValueError(f"Unexpected dataset override: {ds.dataset_dir}")
    manifest = json.loads((ds.dataset_dir / "manifest.json").read_text())
    raw_fps = float(manifest["fps"])
    video_fps = raw_fps / ds.video_stride
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_err = []
    all_pred = []
    all_target = []
    episode_results = []
    valid = np.arange(24) < 22

    for episode in range(len(ds._episodes)):
        starts = [s for ep, s in ds._windows if ep == episode and s % args.chunk_stride == 0]
        starts = starts[: int(args.max_windows)]
        if not starts:
            print(f"[eval] episode={episode} has no valid windows; skipping", flush=True)
            continue
        print(f"[eval] episode={episode:02d} windows={len(starts)}", flush=True)
        ep_out = out_root / f"episode_{episode:02d}"
        ep_out.mkdir(parents=True, exist_ok=True)
        writers = None
        if args.decode_video:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw

            video_dir = ep_out / "video"
            video_dir.mkdir(parents=True, exist_ok=True)
            writers = ExitStack()
            atexit.register(writers.close)
            opts = dict(
                fps=video_fps,
                codec="libx264",
                macro_block_size=1,
                ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
            comparison = writers.enter_context(
                imageio.get_writer(str(video_dir / "gt_vs_generated_all_chunks.mp4"), **opts)
            )
            generated = writers.enter_context(imageio.get_writer(str(video_dir / "generated_all_chunks.mp4"), **opts))

        predictions, targets, latencies = [], [], []
        for i, start in enumerate(starts, 1):
            item = (episode, start)
            sample = ds[ds._windows.index(item)]
            table = ds._read_table(episode).slice(start, ds.num_frames)
            raw_state = np.asarray(table["observation.state"][0].as_py(), dtype=np.float32)
            raw_target = np.asarray(table["action"].to_pylist(), dtype=np.float32)[: ds.num_frames - 1]
            t0 = time.perf_counter()
            result = engine.generate(
                {
                    "prompt": sample["prompt"],
                    "first_frame_image": sample["first_frame_image"],
                    "proprio": raw_state,
                    "seed": int(args.seed),
                    "denoise_steps": int(args.denoise_steps),
                }
            )
            latency = (time.perf_counter() - t0) * 1000.0
            pred = np.asarray(result["actions"], dtype=np.float32)
            if pred.shape != raw_target.shape:
                raise RuntimeError(f"shape mismatch episode={episode} start={start}: {pred.shape} vs {raw_target.shape}")
            predictions.append(pred)
            targets.append(raw_target)
            latencies.append(latency)
            if args.decode_video:
                video = result.get("video")
                if video is None or len(video) != len(sample["video"]):
                    raise RuntimeError("Generated video missing or frame count differs from GT")
                for j, (gt, frame) in enumerate(zip(sample["video"], video)):
                    if args.chunk_stride == ds.num_frames - 1 and i > 1 and j == 0:
                        continue
                    frame = frame.convert("RGB")
                    if frame.size != gt.size:
                        raise RuntimeError(f"Generated/GT size mismatch: {frame.size}, {gt.size}")
                    w, h = frame.size
                    panel = Image.new("RGB", (w * 2, h + 56), "black")
                    panel.paste(gt, (0, 56))
                    panel.paste(frame, (w, 56))
                    draw = ImageDraw.Draw(panel)
                    raw_frame = start + j * ds.video_stride
                    draw.text((8, 5), "GT (recorded)", fill="white")
                    draw.text((w + 8, 5), "Generated (first frame conditioned)", fill="white")
                    draw.text((8, 23), f"Episode {episode} | chunk {i}/{len(starts)} | start={start} | frame={raw_frame}", fill="white")
                    draw.text((8, 39), f"Non-overlap chunks | video {video_fps:g} fps | denoise={args.denoise_steps}", fill="white")
                    comparison.append_data(np.asarray(panel))
                    generated.append_data(np.asarray(frame))
            err = pred[:, valid] - raw_target[:, valid]
            print(
                f"[eval] episode={episode:02d} {i:02d}/{len(starts)} start={start:03d} "
                f"latency={latency/1000:.2f}s mae={np.abs(err).mean():.5f}",
                flush=True,
            )

        if writers is not None:
            writers.close()
            atexit.unregister(writers.close)
            (ep_out / "video" / "manifest.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(args.checkpoint_dir),
                        "dataset_dir": str(args.dataset_dir),
                        "episode": episode,
                        "chunk_stride": args.chunk_stride,
                        "video_fps": video_fps,
                        "raw_fps": raw_fps,
                        "seed": args.seed,
                        "denoise_steps": args.denoise_steps,
                        "dit_cache_enabled": bool(deploy_cfg.optimization.dit_cache.enabled),
                        "description": "Each chunk uses its recorded first image and state, not autoregressive rollout. Shared boundary frames removed for contiguous chunks.",
                        "last_covered_raw_frame": starts[-1] + ds.num_frames - 1,
                        "omitted_tail_frames": int(ds._episodes[episode]["length"]) - (starts[-1] + ds.num_frames),
                        "chunks": starts,
                    },
                    indent=2,
                )
            )
        pred_arr = np.stack(predictions)
        target_arr = np.stack(targets)
        err = pred_arr[:, :, valid] - target_arr[:, :, valid]
        metrics = {
            "checkpoint_dir": str(args.checkpoint_dir),
            "dataset_dir": str(args.dataset_dir),
            "episode": episode,
            "windows": len(starts),
            "denoise_steps": args.denoise_steps,
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "max_abs_error": float(np.abs(err).max()),
            "per_dim_mae": np.abs(err).mean(axis=(0, 1)).tolist(),
            "latency_ms_mean": float(np.mean(latencies)),
        }
        (ep_out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        np.savez_compressed(ep_out / "metrics.npz", predictions=pred_arr, targets=target_arr, starts=np.asarray(starts))
        episode_results.append(metrics)
        all_err.append(err)
        all_pred.append(pred_arr)
        all_target.append(target_arr)

    joined_err = np.concatenate(all_err, axis=0)
    aggregate = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_dir": str(args.dataset_dir),
        "episodes": len(episode_results),
        "windows": int(sum(x["windows"] for x in episode_results)),
        "denoise_steps": args.denoise_steps,
        "chunk_stride": args.chunk_stride,
        "seed": args.seed,
        "valid_action_dims": 22,
        "dit_cache_enabled": bool(deploy_cfg.optimization.dit_cache.enabled),
        "mae": float(np.abs(joined_err).mean()),
        "rmse": float(np.sqrt(np.mean(joined_err**2))),
        "max_abs_error": float(np.abs(joined_err).max()),
        "per_dim_mae": np.abs(joined_err).mean(axis=(0, 1)).tolist(),
        "episode_metrics": episode_results,
    }
    (out_root / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

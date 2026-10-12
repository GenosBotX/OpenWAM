"""Offline Linglong train-set inference/evaluation for an OpenWAM checkpoint.

This evaluates observation windows without teacher forcing: only the first
multiview image, the raw 24-D measured state, and the task prompt are passed to
the policy.  The generated 80-D unified action trajectory is returned through
the checkpoint normalizer and compared with the raw 24-D ``q_exp`` targets.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--max-windows", type=int, default=138)
    p.add_argument("--chunk-stride", type=int, default=None,
                   help="Window start stride for episode rollout comparison. Use 32 for non-overlapping 32-action chunks.")
    p.add_argument("--denoise-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--decode-video", action="store_true", help="Decode and save generated Wan video frames.")
    p.add_argument("--video-output", default=None, help="Directory for generated and GT comparison MP4s.")
    p.add_argument("--output", required=True)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(repo))
    from openwam.dataloader.linglong import LinglongDataset
    from openwam.deploy.engine import JointInferenceEngine
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.deploy.server import _load_deploy_yaml, merge_deploy_cfg

    ckpt = Path(args.checkpoint_dir)
    print(f"[eval] loading checkpoint {ckpt}", flush=True)
    cfg, architecture = load_from_checkpoint_dir(str(ckpt), device="cuda:0")
    deploy_cfg = _load_deploy_yaml()
    # Keep evaluation deterministic and avoid compile warmup for this offline
    # measurement.  The checkpoint's train-time model/dataloader config wins.
    deploy_cfg.inference.denoise_steps = int(args.denoise_steps)
    deploy_cfg.optimization.compile.enabled = False
    deploy_cfg.optimization.decode_video = bool(args.decode_video)
    cfg = merge_deploy_cfg(cfg, deploy_cfg)
    engine = JointInferenceEngine(cfg=cfg, architecture=architecture)

    dl_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataloader, resolve=True))
    dl_cfg.dataset_dir = args.dataset_dir
    dl_cfg.overfit_single = False
    dl_cfg.overfit_repeats = 1
    ds = LinglongDataset.from_config(dl_cfg, split="train")
    episode_windows = [x for x in ds._windows if x[0] == int(args.episode)]
    if not episode_windows:
        raise ValueError(f"episode {args.episode} has no valid windows")
    if args.chunk_stride is not None:
        if args.chunk_stride <= 0:
            raise ValueError("chunk-stride must be positive")
        wanted = set(range(0, int(args.max_windows) * int(args.chunk_stride), int(args.chunk_stride)))
        episode_windows = [x for x in episode_windows if x[1] in wanted]
    else:
        episode_windows = episode_windows[: int(args.max_windows)]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(args.dataset_dir) / "manifest.json").read_text())
    raw_fps = float(manifest["fps"])
    video_fps = raw_fps / ds.video_stride
    video_dir = Path(args.video_output or (str(out.parent) + "/video"))
    video_manifest = []
    output_frame_count = 0
    contiguous = args.chunk_stride == ds.num_frames - 1
    if args.decode_video:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw

        video_dir.mkdir(parents=True, exist_ok=True)
        writers = ExitStack()
        options = dict(fps=video_fps, codec="libx264", macro_block_size=1,
                       ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        comparison = writers.enter_context(imageio.get_writer(str(video_dir / "gt_vs_generated_all_chunks.mp4"), **options))
        generated = writers.enter_context(imageio.get_writer(str(video_dir / "generated_all_chunks.mp4"), **options))
        # Also close encoders on an exception so already-generated frames remain recoverable.
        import atexit
        atexit.register(writers.close)
    predictions = []
    targets = []
    latencies = []
    starts = []
    print(f"[eval] windows={len(episode_windows)} denoise_steps={args.denoise_steps}", flush=True)
    for i, (local, start) in enumerate(episode_windows, 1):
        sample = ds[ds._windows.index((local, start))]
        table = ds._read_table(local).slice(start, ds.num_frames)
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
        if args.decode_video:
            video = result.get("video")
            if video is None or len(video) != len(sample["video"]):
                raise RuntimeError("Generated video missing or frame count differs from GT")
            for j, (gt, frame) in enumerate(zip(sample["video"], video)):
                # Consecutive 32-action chunks share one boundary observation
                # frame (start_{k+1}=start_k+32). Do not duplicate it in the
                # episode-level video stream.
                if contiguous and i > 1 and j == 0:
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
                draw.text((8, 23), f"Episode {args.episode} | chunk {i}/{len(episode_windows)} | start={start} | frame={raw_frame} | t={raw_frame/raw_fps:.3f}s", fill="white")
                layout = "Non-overlap action chunks" if contiguous else "Window comparison"
                draw.text((8, 39), f"{layout} | video {video_fps:g} fps | denoise={args.denoise_steps} | seed={args.seed}", fill="white")
                comparison.append_data(np.asarray(panel))
                generated.append_data(np.asarray(frame))
            skip = int(contiguous and i > 1)
            count = len(video) - skip
            video_manifest.append({"chunk": i, "start_frame": start,
                                   "raw_frame_indices": list(range(start, start + ds.num_frames, ds.video_stride))[skip:],
                                   "output_frame_start": output_frame_count,
                                   "output_frame_count": count})
            output_frame_count += count
        latency = (time.perf_counter() - t0) * 1000.0
        pred = np.asarray(result["actions"], dtype=np.float32)
        if pred.shape != raw_target.shape:
            raise RuntimeError(f"shape mismatch at window {start}: pred={pred.shape} target={raw_target.shape}")
        predictions.append(pred)
        targets.append(raw_target)
        latencies.append(latency)
        starts.append(int(start))
        valid = np.arange(raw_target.shape[-1]) < 22
        err = np.abs(pred[:, valid] - raw_target[:, valid])
        print(
            f"[eval] {i:03d}/{len(episode_windows)} start={start:03d} "
            f"latency={latency/1000:.2f}s mae={err.mean():.5f} max={err.max():.5f}",
            flush=True,
        )

    if args.decode_video:
        writers.close()
        atexit.unregister(writers.close)
        (video_dir / "manifest.json").write_text(json.dumps({
            "checkpoint": str(ckpt), "episode": args.episode, "seed": args.seed,
            "denoise_steps": args.denoise_steps, "raw_fps": raw_fps, "video_fps": video_fps,
            "dit_cache_enabled": bool(deploy_cfg.optimization.dit_cache.enabled),
            "description": "Each chunk is conditioned on its recorded first image and state. Not an autoregressive rollout. Subsequent GT frames are used for comparison only. Shared boundary frames are removed for contiguous chunks.",
            "chunk_stride": args.chunk_stride,
            "episode_frames": int(ds._episodes[args.episode]["length"]),
            "last_covered_raw_frame": int(episode_windows[-1][1] + ds.num_frames - 1),
            "omitted_tail_frames": int(ds._episodes[args.episode]["length"]) - (episode_windows[-1][1] + ds.num_frames),
            "chunks": video_manifest,
        }, indent=2))
        print(f"[eval] MP4 files finalized in {video_dir}", flush=True)
    pred = np.stack(predictions)
    target = np.stack(targets)
    valid = np.arange(target.shape[-1]) < 22
    err = pred[:, :, valid] - target[:, :, valid]
    metrics = {
        "checkpoint_dir": str(ckpt),
        "dataset_dir": str(args.dataset_dir),
        "episode": int(args.episode),
        "windows": int(len(episode_windows)),
        "denoise_steps": int(args.denoise_steps),
        "action_dim": int(pred.shape[-1]),
        "valid_action_dims": 22,
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs_error": float(np.abs(err).max()),
        "per_dim_mae": np.abs(err).mean(axis=(0, 1)).tolist(),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
    }
    np.savez_compressed(out.with_suffix(".npz"), predictions=pred, targets=target, starts=np.asarray(starts))
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

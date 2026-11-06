"""Load one RTC model, compare cache on/off on recorded requests, no robot IO."""
import argparse
from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'third_party'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt-dir', required=True)
    parser.add_argument('--ckpt-name', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--request-json', required=True, nargs='+', type=Path,
                        help='Recorded wire obs JSONs with encoded images/state/prompt/action_prefix')
    parser.add_argument('--prefix-lengths', nargs='+', type=int, default=None,
                        help='Truncate recorded prefix to these d; default keeps recorded d. Never pads.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--cosine-threshold', type=float, default=.99)
    parser.add_argument('--max-skips', type=int, default=3)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('overrides', nargs='*', help='Same OmegaConf dotlist overrides as scripts/deploy.py')
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or not 0 < args.cosine_threshold <= 1 or args.max_skips < 0:
        parser.error('invalid repeats/warmup/cache settings')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO)

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from openwam.deploy.optimizations.dit_cache import DiTVelocityCache
    from openwam.deploy.rtc_cache_eval import paired_run
    from openwam.deploy.server import _load_deploy_yaml, build_server_from_config

    cfg = _load_deploy_yaml(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    OmegaConf.update(cfg, 'inference.action_response_mode', 'chunk', merge=False)
    OmegaConf.update(cfg, 'inference.inference_mode', 'sync', merge=False)
    OmegaConf.update(cfg, 'optimization.dit_cache.enabled', False, merge=False)
    server = build_server_from_config(cfg, args.ckpt_dir, device=args.device, ckpt_name=args.ckpt_name)
    cache = DiTVelocityCache(cosine_threshold=args.cosine_threshold, max_consecutive_skips=args.max_skips)
    synchronize = (lambda: torch.cuda.synchronize(args.device)) if str(args.device).startswith('cuda') else (lambda: None)
    rows = []
    try:
        server._init_policy()
        engine = server.engine
        trained = getattr(engine.architecture, 'rtc_prefix_lengths', ())
        if not trained:
            raise ValueError('an RTC-trained checkpoint is required')
        if engine._cfg_scale > 1:
            raise ValueError('CFG scale > 1 disables the existing cache; use the deployment CFG scale=1')
        try:
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip()
        except (OSError, subprocess.SubprocessError):
            revision = None
        manifest = dict(arguments={**vars(args), 'request_json': list(map(str, args.request_json)),
                                   'output_dir': str(args.output_dir)}, revision=revision,
                        torch=torch.__version__, cuda=torch.version.cuda,
                        gpu=torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else None,
                        resolved_config=OmegaConf.to_container(server.cfg, resolve=True),
                        interpretation='Physical suffix deviation from cache-off with identical weights/seed. Not task accuracy.')
        (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str)+'\n')
        for sample_id, path in enumerate(args.request_json):
            raw = path.read_bytes()
            obs = json.loads(raw)
            prefix = obs.get('action_prefix', [])
            lengths = args.prefix_lengths if args.prefix_lengths is not None else [len(prefix)]
            if any(d not in trained or d > len(prefix) for d in lengths):
                raise ValueError(f'{path}: requested d must be trained and covered by its recorded prefix ({len(prefix)})')
            for d in lengths:
                request = deepcopy(obs)
                request['action_prefix'] = prefix[:d]
                processed = server._obs_preprocessor.preprocess(request)
                conditions = server._policy._build_conditions(processed)
                for seed in args.seeds:
                    conditions['seed'] = seed
                    with torch.inference_mode():
                        for _ in range(args.warmup):
                            for selected in (None, cache):
                                engine._dit_cache = selected
                                engine.generate(deepcopy(conditions))
                        for repeat in range(args.repeats):
                            report, actions = paired_run(engine, conditions, cache, synchronize=synchronize,
                                                         cached_first=bool(repeat % 2))
                            report.update(sample_id=sample_id, request_path=str(path),
                                          request_sha256=hashlib.sha256(raw).hexdigest(), repeat=repeat)
                            name = f's{sample_id}_d{d}_seed{seed}_r{repeat}.npz'
                            np.savez_compressed(args.output_dir/name, **actions)
                            report['actions_file'] = name
                            rows.append(report)
                            with (args.output_dir/'results.jsonl').open('a') as stream:
                                stream.write(json.dumps(report)+'\n')
                            print(json.dumps(report), flush=True)
        grouped = {}
        for d in sorted({row['d'] for row in rows}):
            group = [row for row in rows if row['d'] == d]
            grouped[str(d)] = dict(pairs=len(group), latency_s={
                mode: dict(p50=float(np.median([row['latency_s'][mode] for row in group])),
                           p95=float(np.quantile([row['latency_s'][mode] for row in group], .95)))
                for mode in ('off', 'on')},
                total_skips=sum(row['cache_stats']['total_skips'] for row in group),
                total_steps=sum(row['cache_stats']['total_steps'] for row in group),
                worst_suffix_error={arm: {metric: max(row['cache_error'][arm][metric]['max'] for row in group)
                    for metric in ('position_mm', 'rotation_deg', 'gripper_abs')} for arm in ('left', 'right')})
        (args.output_dir/'summary.json').write_text(json.dumps(grouped, indent=2)+'\n')
    finally:
        server.shutdown()


if __name__ == '__main__':
    main()

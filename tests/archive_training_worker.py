"""Small real training loop for archive integration checks; no production dataset or model needed."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import save_file
from openwam.train.openwam_trainer import OpenWAMTrainer
from openwam.train.utils import experiment_archive as archive


class TinyArchitecture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(4, 4)
        self.dtype = torch.float32

    def forward(self, x):
        return self.net(x)

    def get_trainable_modules(self):
        return {'net': self.net}

    def set_dtype_device(self, dtype, device):
        self.to(device=device, dtype=dtype)

    def move_frozen_to_device(self, device):
        pass

    def save_assets_for_deployment(self, output, cfg):
        folder = Path(output) / 'tokenizer'
        folder.mkdir()
        (folder / 'vocab.json').write_text('{"test": 0}')

    def save_checkpoint(self, path, state_dict=None, **kwargs):
        save_file({k: v.cpu().contiguous() for k, v in state_dict.items()}, path)


class TinyTrainer(OpenWAMTrainer):
    def __init__(self, cfg, accelerator, case):
        self.cfg, self.accelerator, self.case = cfg, accelerator, case
        self.architecture = TinyArchitecture()
        self.initial = self.architecture.net.weight.detach().clone()
        self.dataset = [torch.arange(4, dtype=torch.float32) / 4 for _ in range(16)]
        self._rank, self._run_seed, self._ckpt_source_dir = 0, 42, None
        self.calls = 0

    def compute_loss(self, batch):
        self.calls += 1
        if self.calls == 3:
            if self.case == 'exception':
                raise RuntimeError('injected training exception')
            if self.case == 'interrupt':
                raise KeyboardInterrupt('injected training interrupt')
            if self.case == 'sigterm':
                os.kill(os.getpid(), signal.SIGTERM)
        x = torch.stack(batch).to(self.accelerator.device)
        loss = ((self.architecture(x) - x * 0.5) ** 2).mean()
        return {'total': loss, 'video': loss.detach(), 'action': loss.detach()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=['delete', 'keep', 'max_steps', 'exception', 'interrupt', 'sigterm', 'upload_fail'], required=True)
    parser.add_argument('--live-tos', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--archive-path')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    torch.distributed.init_process_group('gloo', init_method='file://' + str(args.output / 'dist_init'), rank=0, world_size=1)
    accelerator = Accelerator(cpu=args.cpu, gradient_accumulation_steps=2)
    options = OmegaConf.load(Path(__file__).resolve().parents[1] / 'configs/training/archive/tos.yaml')
    options.keep_local = args.case == 'keep'
    options.path = args.archive_path
    # Real smoke uses the default date/name layout and a unique experiment name.
    cfg = OmegaConf.create({'training': {
        'output_path': str(args.output / ('archive_training_' + args.case + '_' + args.output.name)),
        'archive': options, 'num_epochs': 1, 'max_steps': 4 if args.case == 'max_steps' else None,
        'batch_size': 2, 'gradient_accumulation_steps': 2, 'dataset_num_workers': 0,
        'learning_rate': 0.05, 'weight_decay': 0.0, 'max_grad_norm': 1.0,
        'lr_scheduler': 'cosine', 'warmup_ratio': 0.0, 'lr_min_ratio': 0.1,
        'save_steps': 2, 'keep_last_k_ckpts': 2, 'save_full_states_for_resume': True,
        'save_trainable_only': False, 'async_checkpoints': True,
        'checkpoint_staging_dir': str(args.output / 'stage'),
    }, 'project': {'seed': 42, 'wandb': {'project': None}}})
    real_archive = archive.archive_run
    store = None
    if not args.live_tos or args.case == 'upload_fail':
        from test_experiment_archive import Store
        store = Store()
        if args.case == 'upload_fail':
            store.fail = '.safetensors'
        archive.time.sleep = lambda _: None
    results = {}
    def upload(*a, **kw):
        (args.output / 'upload_entered').write_text('yes')
        root = Path(a[0])
        before = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in root.rglob('*') if p.is_file()}
        if store is not None:
            kw['client'] = store
        result = real_archive(*a, **kw)
        results.update(result)
        bucket, prefix = result['destination'][6:].split('/', 1)
        if args.live_tos:
            client = archive.make_client(OmegaConf.to_container(options, resolve=True))
            manifest = json.loads(client.get_object(bucket, prefix + 'archive_manifest.json').read())
            restored = args.output / 'restored'
            for row in manifest['files']:
                data = client.get_object(bucket, prefix + row['path']).read()
                assert hashlib.sha256(data).hexdigest() == before[row['path']]
                path = restored / row['path']
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            results['download_sha256_verified'] = True
            results['restore_dir'] = str(restored)
        return result
    archive.archive_run = upload
    trainer = TinyTrainer(cfg, accelerator, args.case)
    error = None
    try:
        trainer.train()
    except (RuntimeError, KeyboardInterrupt) as exc:
        error = str(exc)
    finally:
        root_paths = list(Path(cfg.training.output_path).glob('20*'))
        for root in root_paths:
            archive._close_run_logs(root)
    failed = args.case in ('exception', 'interrupt', 'upload_fail')
    assert bool(error) == failed, error
    if failed:
        assert root_paths and (root_paths[0] / ('checkpoint_step_8.safetensors' if args.case == 'upload_fail' else 'checkpoint_step_2.safetensors')).exists()
        assert not (args.output / 'upload_entered').exists() if args.case != 'upload_fail' else True
    else:
        assert not torch.equal(trainer.initial, trainer.architecture.net.weight.detach().cpu())
        assert bool(root_paths) == (args.case == 'keep')
        assert results['step'] == (4 if args.case == 'max_steps' else 8)
        if args.live_tos:
            expected = {k: v.detach().clone() for k,v in trainer.architecture.state_dict().items()}
            with torch.no_grad():
                trainer.architecture.net.weight.zero_()
            accelerator.load_state(str(Path(results['restore_dir']) / f'accel_state_step_{results["step"]}'))
            assert all(torch.equal(expected[k], v) for k,v in trainer.architecture.state_dict().items())
            assert accelerator._optimizers[0].state_dict()['state']
            results['resume_state_restored'] = True
    results.update(case=args.case, error=error, device=str(accelerator.device), training_steps=trainer.calls,
                   upload_entered=(args.output / 'upload_entered').exists(), local_runs=[str(p) for p in root_paths])
    (args.output / 'result.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()

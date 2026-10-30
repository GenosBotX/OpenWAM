"""Destructive cleanup is gated by complete remote verification."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openwam.train.utils import experiment_archive as archive


class Store:
    def __init__(self):
        self.objects = {}
        self.after_upload = None
        self.head_count = {}
        self.fail = None

    def put_object(self, bucket, key, content, content_length):
        if self.fail and self.fail in key:
            raise IOError('injected upload failure')
        assert content_length == len(content)
        self.objects[(bucket, key)] = content
        return SimpleNamespace(hash_crc64_ecma=self.crc(content))

    @staticmethod
    def crc(content):
        return int(hashlib.sha256(content).hexdigest()[:16], 16)

    def upload_file(self, bucket, key, filename, **kwargs):
        data = Path(filename).read_bytes()
        out = self.put_object(bucket, key, data, len(data))
        if self.after_upload:
            self.after_upload()
        return out

    def head_object(self, bucket, key):
        self.head_count[key] = self.head_count.get(key, 0) + 1
        data = self.objects[(bucket, key)]
        return SimpleNamespace(content_length=len(data), hash_crc64_ecma=self.crc(data))


def run_tree(tmp_path, *, states=True):
    root = tmp_path / 'experiments' / 'run'
    root.mkdir(parents=True)
    (root / 'config.yaml').write_text('training:\n  save_full_states_for_resume: true\n')
    (root / 'normalization_stats.npy').write_bytes(b'stats')
    (root / 'training.log').write_text('loss=0.1')
    (root / 'tokenizer').mkdir()
    (root / 'tokenizer' / 'vocab.json').write_text('{}')
    for step in (9, 10):
        (root / f'checkpoint_step_{step}.safetensors').write_bytes(f'weights {step}'.encode())
        if states:
            state = root / f'accel_state_step_{step}'
            state.mkdir()
            (state / 'optimizer.bin').write_bytes(b'optimizer')
            (state / 'random_states_0.pkl').write_bytes(b'rng')
            (state / 'trainer_state.json').write_text(json.dumps(dict(
                global_step=step, model_checkpoint={'weights': f'checkpoint_step_{step}.safetensors'})))
    return root


def options(tmp_path, **kwargs):
    return dict(root='tos://test/experiments', local_root=str(tmp_path), workers=2, **kwargs)


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(archive.time, 'sleep', lambda _: None)


@pytest.mark.parametrize('keep', [True, False])
def test_latest_pair_assets_and_cleanup(tmp_path, keep):
    root = run_tree(tmp_path)
    sibling = root.parent / 'other_run'
    sibling.mkdir()
    (sibling / 'valuable').write_text('keep')
    store = Store()
    before = archive.snapshot(root)
    result = archive.archive_run(root, options(tmp_path, keep_local=keep), expected_step=10, client=store)
    assert result['step'] == 10
    assert result['local_deleted'] is (not keep)
    assert (sibling / 'valuable').read_text() == 'keep'
    if keep:
        assert archive.snapshot(root) == before  # keep_local also retains older checkpoints
    else:
        assert not root.exists()
    keys = '\n'.join(k for _, k in store.objects)
    assert 'checkpoint_step_10.safetensors' in keys
    assert 'accel_state_step_10/optimizer.bin' in keys
    assert 'checkpoint_step_9.safetensors' not in keys
    assert 'accel_state_step_9' not in keys
    assert 'normalization_stats.npy' in keys and 'tokenizer/vocab.json' in keys
    assert 'ARCHIVE_COMPLETE.json' in keys and 'LATEST.json' in keys
    for k, count in store.head_count.items():
        if k.endswith('.safetensors'):
            assert count == 2


@pytest.mark.parametrize('failure', ['weights', 'manifest', 'completion', 'pointer', 'second_head'])
def test_failures_preserve_every_local_file(tmp_path, failure):
    root = run_tree(tmp_path)
    before = archive.snapshot(root)
    store = Store()
    store.fail = {'weights': 'checkpoint_step_10', 'manifest': 'archive_manifest.json',
                  'completion': 'ARCHIVE_COMPLETE.json', 'pointer': 'LATEST.json'}.get(failure)
    if failure == 'second_head':
        head = store.head_object
        def corrupt(bucket, key):
            result = head(bucket, key)
            if key.endswith('optimizer.bin') and store.head_count[key] > 1:
                result.hash_crc64_ecma = -1
            return result
        store.head_object = corrupt
    with pytest.raises(RuntimeError, match='本地完整保留'):
        archive.archive_run(root, options(tmp_path), client=store)
    assert archive.snapshot(root) == before


def test_source_modified_during_upload_is_not_deleted(tmp_path):
    root = run_tree(tmp_path)
    store = Store()
    store.after_upload = lambda: (root / 'new.log').write_text('new')
    with pytest.raises(RuntimeError, match='本地完整保留'):
        archive.archive_run(root, options(tmp_path), client=store)
    assert (root / 'new.log').exists()
    assert (root / 'checkpoint_step_9.safetensors').exists()
    assert not any(k.endswith('ARCHIVE_COMPLETE.json') for _, k in store.objects)


@pytest.mark.parametrize('case', ['missing', 'mismatched_step', 'empty', 'bad_pair', 'final_step'])
def test_incomplete_latest_never_falls_back_to_old(tmp_path, case):
    root = run_tree(tmp_path)
    p = root / 'accel_state_step_10' / 'trainer_state.json'
    if case == 'missing':
        p.unlink()
    elif case == 'mismatched_step':
        p.write_text('{"global_step": 9}')
    elif case == 'bad_pair':
        p.write_text('{"global_step": 10, "model_checkpoint": {"weights": "checkpoint_step_9.safetensors"}}')
    elif case == 'empty':
        (p.parent / 'optimizer.bin').unlink()
        (p.parent / 'random_states_0.pkl').unlink()
    before = archive.snapshot(root)
    store = Store()
    with pytest.raises((ValueError, FileNotFoundError)):
        archive.archive_run(root, options(tmp_path), client=store, expected_step=11 if case == 'final_step' else 10)
    assert archive.snapshot(root) == before
    assert store.objects == {}


def test_weights_only_mode(tmp_path):
    root = run_tree(tmp_path, states=False)
    result = archive.archive_run(root, options(tmp_path), client=Store(), require_state=False)
    assert result['local_deleted']


def test_symlink_does_not_delete_target(tmp_path):
    root = run_tree(tmp_path)
    target = tmp_path / 'outside'
    target.write_text('important')
    (root / 'linked').symlink_to(target)
    with pytest.raises(ValueError, match='符号链接'):
        archive.archive_run(root, options(tmp_path), client=Store())
    assert target.read_text() == 'important'
    assert root.exists()


def test_retry_generation_does_not_overwrite_verified_archive(tmp_path):
    root = run_tree(tmp_path)
    store = Store()
    first = archive.archive_run(root, options(tmp_path, keep_local=True), client=store)
    saved = dict(store.objects)
    store.fail = 'optimizer.bin'
    with pytest.raises(RuntimeError):
        archive.archive_run(root, options(tmp_path), client=store)
    assert root.exists()
    assert all(store.objects[k] == v for k, v in saved.items())
    assert first['destination']


def test_logs_closed_before_snapshot(tmp_path):
    root = run_tree(tmp_path)
    archive.attach_training_log(root)
    archive._close_run_logs(root)
    before = (root / 'training.log').read_bytes()
    archive.logger.warning('after close')
    assert (root / 'training.log').read_bytes() == before


def test_refuse_local_root(tmp_path):
    root = run_tree(tmp_path)
    with pytest.raises(ValueError, match='local_root'):
        archive.archive_run(root, options(root), client=Store())
    assert root.exists()


@pytest.mark.parametrize('save_steps,enabled', [(10, True), (0, True), (10, False)])
def test_trainer_archives_after_final_save_and_log_finish(tmp_path, monkeypatch, save_steps, enabled):
    from omegaconf import OmegaConf
    from openwam.train import openwam_trainer as module
    events = []
    trainer = object.__new__(module.OpenWAMTrainer)
    trainer.cfg = OmegaConf.create({'training': {'save_full_states_for_resume': True,
                                                'archive': {'enabled': enabled}}})
    trainer.accelerator = SimpleNamespace(sync_gradients=True, wait_for_everyone=lambda: events.append('barrier'))
    trainer.publish_checkpoint = lambda *a, **k: events.append('save')
    trainer._ckpt_publisher = SimpleNamespace(wait=lambda: events.append('save_done'))
    # Older common Trainer saves synchronously.
    trainer.architecture = None
    trainer.save_resumable_state = lambda *a: events.append('state')
    if hasattr(module, 'save_weights'):
        monkeypatch.setattr(module, 'save_weights', lambda *a, **k: events.append('save'))
        monkeypatch.setattr(module, 'manage_checkpoints', lambda *a: events.append('save_done'))
    monkeypatch.setattr(archive, 'finish_archive', lambda *a, **k: events.append('archive'))
    trainer.finish_training(str(tmp_path), 10, save_steps, 1, True,
                            SimpleNamespace(finish=lambda: events.append('wandb_done')))
    if save_steps and enabled:
        assert events.index('save') < events.index('save_done') < events.index('wandb_done') < events.index('archive')
    else:
        assert 'archive' not in events


@pytest.mark.parametrize('failure', [False, True])
def test_distributed_only_rank_zero_uploads_and_propagates_error(tmp_path, failure):
    import os
    import subprocess
    import sys
    worker = tmp_path / 'worker.py'
    worker.write_text('''
import json
import sys
from pathlib import Path
from accelerate import Accelerator
from omegaconf import OmegaConf
from openwam.train.utils import experiment_archive as archive
accelerator = Accelerator(cpu=True)
root = Path(sys.argv[1])
fail = sys.argv[2] == 'True'
def upload(*args, **kwargs):
    with (root / 'uploads.txt').open('a') as f:
        f.write(str(accelerator.process_index) + '\\n')
    if fail:
        raise RuntimeError('injected upload error')
    return {'ok': True}
archive.archive_run = upload
try:
    archive.finish_archive(root / 'run', OmegaConf.create({'enabled':True, 'wait_timeout_s':30}), accelerator, step=10, require_state=True)
    result = 'ok'
except Exception as exc:
    result = str(exc)
(root / ('rank_' + str(accelerator.process_index))).write_text(result)
''')
    env = dict(os.environ, ACCELERATE_USE_CPU='true', OMP_NUM_THREADS='1', PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2', str(worker), str(tmp_path), str(failure)],
                            env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'uploads.txt').read_text() == '0\n'
    for rank in (0, 1):
        message = (tmp_path / f'rank_{rank}').read_text()
        assert ('injected upload error' in message) if failure else message == 'ok'


def test_archive_config_inheritance_and_cli_overrides():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    config_dir = Path(__file__).resolve().parents[1] / 'configs'
    names = ['train', 'train_linglong_0923_1010', 'train_linglong_tos',
             'train_piper_x_20261022_20261028_relative_eef_16gpu_e5',
             'train_piper_x_20261022_20261028_egodex_pitch20_relative_eef_16gpu_e5',
             'train_egodex_linglong_joint_16gpu_bs6']
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        for name in names:
            if not (config_dir / (name + '.yaml')).exists():
                continue
            cfg = compose(config_name=name)
            assert cfg.training.archive.enabled is True, name
            assert cfg.training.archive.keep_local is False, name
            assert cfg.training.save_full_states_for_resume is True, name
            overridden = compose(config_name=name, overrides=['training.archive.keep_local=true'])
            assert overridden.training.archive.keep_local is True, name
            disabled = compose(config_name=name, overrides=['training.archive.enabled=false'])
            assert disabled.training.archive.enabled is False, name


@pytest.mark.parametrize('custom,expected', [
    (None, ('test', 'experiments/2026-10-30/piper_relative')),
    ('tangyuting/piper/run_01', ('test', 'experiments/tangyuting/piper/run_01')),
    ('tos://other/custom/run', ('other', 'custom/run')),
])
def test_destination_date_name_and_override(tmp_path, custom, expected):
    root = tmp_path / 'piper_relative' / '2026-10-30_12-00-00'
    assert archive.destination_for(root, options(tmp_path, path=custom)) == expected


@pytest.mark.parametrize('custom', ['../other', '/absolute', 'foo/../bar', 'foo?query=1'])
def test_destination_rejects_unsafe_path(tmp_path, custom):
    with pytest.raises(ValueError):
        archive.destination_for(tmp_path / 'exp' / 'run', options(tmp_path, path=custom))


@pytest.mark.parametrize('case', ['delete', 'keep', 'max_steps', 'exception', 'interrupt', 'sigterm', 'upload_fail'])
def test_real_training_loop_archive_lifecycle(tmp_path, case):
    import os
    import signal
    import subprocess
    import sys
    worker = Path(__file__).with_name('archive_training_worker.py')
    result = subprocess.run([sys.executable, str(worker), '--cpu', '--case', case, '--output', str(tmp_path)],
                            env=dict(os.environ, OMP_NUM_THREADS='1'), capture_output=True, text=True, timeout=90)
    if case == 'sigterm':
        assert result.returncode == -signal.SIGTERM, result.stdout + result.stderr
        assert not (tmp_path / 'upload_entered').exists()
        assert list(tmp_path.glob('archive_training_*/*/checkpoint_step_2.safetensors'))
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        receipt = json.loads((tmp_path / 'result.json').read_text())
        assert receipt['upload_entered'] is (case not in ('exception', 'interrupt'))


@pytest.mark.parametrize('saved', [False, True])
def test_retry_cli_preserves_saved_destination(tmp_path, monkeypatch, saved):
    import runpy
    import sys
    root = run_tree(tmp_path)
    if saved:
        (root / 'config.yaml').write_text('training:\n  archive:\n    root: tos://test/experiments\n    path: custom/piper\n    keep_local: true\n')
    calls = []
    monkeypatch.setattr(archive, 'archive_run', lambda path, options, **kw: calls.append(options) or {})
    monkeypatch.setattr(sys, 'argv', ['archive_experiment.py', '--run-dir', str(root)])
    script = Path(__file__).resolve().parents[1] / 'scripts/archive_experiment.py'
    runpy.run_path(str(script), run_name='__main__')
    if saved:
        assert calls[0]['root'] == 'tos://test/experiments'
        assert calls[0]['path'] == 'custom/piper'
        assert calls[0]['keep_local'] is True
    else:
        assert calls[0].get('path') is None

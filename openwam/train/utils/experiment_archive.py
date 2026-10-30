"""Archive a completed run to TOS, verify it, then optionally remove the local run."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import stat
import time
import uuid

logger = logging.getLogger(__name__)
_WEIGHT = re.compile(r"checkpoint_step_(\d+)\.safetensors$")
_STATE = re.compile(r"accel_state_step_(\d+)$")


def attach_training_log(output_path):
    """Keep rank-zero training diagnostics beside the deploy/resume artifacts."""
    path = Path(output_path).resolve() / 'training.log'
    root_logger = logging.getLogger()
    if any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path
           for h in root_logger.handlers):
        return
    handler = logging.FileHandler(path, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    root_logger.addHandler(handler)


def _close_run_logs(output_path):
    # Do not allow an open FileHandler to append after the upload snapshot.
    root = Path(output_path).resolve()
    loggers = [logging.getLogger()] + [x for x in logging.Logger.manager.loggerDict.values()
                                     if isinstance(x, logging.Logger)]
    for log in loggers:
        for handler in list(log.handlers):
            if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve().is_relative_to(root):
                handler.flush()
                log.removeHandler(handler)
                handler.close()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def snapshot(root):
    """Never follow symlinks or traverse a different filesystem during cleanup."""
    root = Path(root)
    entries = {}
    for folder, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            p = Path(folder) / name
            s = p.lstat()
            if not (stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)) or (stat.S_ISDIR(s.st_mode) and os.path.ismount(p)):
                raise ValueError(f'归档拒绝符号链接、特殊文件或挂载点，保留本地: {p}')
            entries[p.relative_to(root).as_posix()] = dict(
                size=s.st_size if stat.S_ISREG(s.st_mode) else 0,
                mtime_ns=s.st_mtime_ns if stat.S_ISREG(s.st_mode) else 0,
                inode=s.st_ino, mode=s.st_mode, device=s.st_dev,
                directory=stat.S_ISDIR(s.st_mode),
            )
    return entries


def select_files(root, entries, *, expected_step=None, require_state=True):
    root = Path(root)
    steps = [int(m[1]) for name in entries if (m := _WEIGHT.fullmatch(name))]
    if not steps:
        raise ValueError('没有完整 checkpoint；保留本地实验')
    latest = max(steps)
    if expected_step is not None and latest != expected_step:
        raise ValueError(f'最终 step {expected_step} 与最新 checkpoint {latest} 不一致')
    state_name = f'accel_state_step_{latest}'
    if require_state or state_name in entries:
        marker = root / state_name / 'trainer_state.json'
        meta = json.loads(marker.read_text())
        if int(meta['global_step']) != latest:
            raise ValueError('checkpoint 与训练 state 的 step 不一致')
        if not any(n.startswith(state_name + '/') and not r['directory']
                   and n != state_name + '/trainer_state.json' for n, r in entries.items()):
            raise ValueError('训练 state 只有完成标记，没有恢复数据')
        paired = meta.get('model_checkpoint')
        if paired and paired.get('weights') != f'checkpoint_step_{latest}.safetensors':
            raise ValueError('精简 checkpoint 与恢复 state 的配对不一致')
    if 'config.yaml' not in entries:
        raise ValueError('实验缺少 config.yaml')
    selected = []
    for name, rec in entries.items():
        if rec['directory']:
            continue
        top = name.split('/')[0]
        weight, state_dir = _WEIGHT.fullmatch(top), _STATE.fullmatch(top)
        if weight and int(weight[1]) != latest:
            continue
        if state_dir and int(state_dir[1]) != latest:
            continue
        if top in ('.ckpt_partial', '.ckpt_copy_status'):
            continue
        selected.append(name)
    return latest, sorted(selected)


def make_client(options):
    # Reuse the launcher's credential validation; no credential enters a manifest.
    from scripts.run_with_tos_credentials import load_credentials
    import tos

    path = options.get('credentials_file') or os.environ.get('OPENWAM_TOS_CREDENTIALS') or os.environ.get(
        'TOS_CREDENTIAL_FILE', '~/.config/openwam/tos.json')
    load_credentials(path)
    return tos.TosClientV2(
        ak=os.environ['TOS_ACCESS_KEY'], sk=os.environ['TOS_SECRET_KEY'],
        endpoint=options.get('endpoint') or os.environ.get('TOS_ENDPOINT', 'tos-cn-beijing.ivolces.com'),
        region=os.environ.get('TOS_REGION', 'cn-beijing'),
        security_token=os.environ.get('TOS_SECURITY_TOKEN'), enable_crc=True,
        max_connections=32, request_timeout=120, socket_timeout=120,
    )


def destination_for(root, options):
    """Default: root / experiment start date / experiment name; path overrides the suffix."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from openwam.dataloader.tos_cache import parse_tos_uri

    local_root = Path(options.get('local_root', '/vePFS/tangyuting')).expanduser().resolve()
    if root == local_root:
        raise ValueError('不能归档 local_root 本身')
    base = options.get('root') or os.environ.get('OPENWAM_EXPERIMENTS_TOS',
        'tos://ml-platform-auto-created-required-2109640594-cn-beijing/experiments')
    custom = options.get('path')
    if custom:
        custom = str(custom)
        if custom.startswith('tos://'):
            destination = custom
        else:
            if custom.startswith('/') or "\\" in custom or ':' in custom:
                raise ValueError('归档 path 必须为相对路径或完整 tos:// URI')
            destination = str(base).rstrip('/') + '/' + custom
    else:
        # Timestamped run directories retain their start date when retrying on a later day.
        match = re.match(r'^(\d{4}-\d{2}-\d{2})_', root.name)
        date = (datetime.strptime(match[1], '%Y-%m-%d').strftime('%Y-%m-%d') if match
                else datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d'))
        destination = str(base).rstrip('/') + '/' + date + '/' + root.parent.name
    bucket, prefix = parse_tos_uri(destination)
    if not prefix:
        raise ValueError('归档目标必须包含目录前缀')
    return bucket, prefix


def _check_head(client, bucket, key, size, crc):
    h = client.head_object(bucket, key)
    if crc is None or h.content_length != size or str(h.hash_crc64_ecma) != str(crc):
        raise RuntimeError(f'TOS 大小/CRC64 校验失败: {key}')


def _put_json(client, bucket, key, value):
    data = json.dumps(value, ensure_ascii=False, indent=2).encode()
    result = client.put_object(bucket, key, content=data, content_length=len(data))
    _check_head(client, bucket, key, len(data), result.hash_crc64_ecma)


def archive_run(output_path, options, *, expected_step=None, require_state=True, client=None):
    """All upload/verification failures precede any local unlink. Returns a durable receipt."""
    raw = Path(output_path).expanduser().absolute()
    if raw.is_symlink():
        raise ValueError('不能清理符号链接指向的实验目录')
    root = raw.resolve(strict=True)
    if root == Path('/') or (root / '.git').exists():
        raise ValueError('不能归档根目录或代码仓库')
    entries = snapshot(root)
    step, selected = select_files(root, entries, expected_step=expected_step, require_state=require_state)
    bucket, prefix = destination_for(root, options)
    # Control files and SDK multipart checkpoints must outlive the deleted run.
    control = root.parent / '.tos_archives' / root.name
    control.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (control / 'archive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        client = client or make_client(options)
        session = uuid.uuid4().hex
        # Immutable generation: retries never invalidate an earlier verified archive.
        prefix += '/archives/' + session
        manifest = dict(version=1, source=str(root), step=step,
                        destination=f'tos://{bucket}/{prefix}/',
                        selected=selected, excluded=[n for n, r in entries.items()
                                                     if not r['directory'] and n not in selected],
                        source_snapshot=entries, files=[])
        _atomic_json(control / 'pending.json', manifest)

        def transfer(name):
            path = root / name
            for attempt in range(3):
                try:
                    result = client.upload_file(
                        bucket, prefix + '/' + name, str(path),
                        part_size=64 * 1024**2, task_num=4, enable_checkpoint=True,
                        checkpoint_file=str(control / hashlib.sha256(name.encode()).hexdigest()),
                    )
                    _check_head(client, bucket, prefix + '/' + name,
                                entries[name]['size'], result.hash_crc64_ecma)
                    return dict(path=name, size=entries[name]['size'], crc64=str(result.hash_crc64_ecma))
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)

        try:
            with ThreadPoolExecutor(max_workers=int(options.get('workers', 4))) as pool:
                futures = [pool.submit(transfer, name) for name in
                           sorted(selected, key=lambda n: -entries[n]['size'])]
                for future in as_completed(futures):
                    row = future.result()
                    manifest['files'].append(row)
                    logger.info('[archive] %s: %d/%d files', root.name, len(manifest['files']), len(selected))
            # Re-check every remote object after the entire upload, not only each PUT.
            for row in manifest['files']:
                _check_head(client, bucket, prefix + '/' + row['path'], row['size'], row['crc64'])
            if snapshot(root) != entries:
                raise RuntimeError('归档过程中本地实验发生修改；保留所有本地文件')
            manifest['verified_at'] = time.time()
            manifest['bytes'] = sum(row['size'] for row in manifest['files'])
            _put_json(client, bucket, prefix + '/archive_manifest.json', manifest)
            _put_json(client, bucket, prefix + '/ARCHIVE_COMPLETE.json',
                      dict(step=step, files=len(selected), bytes=manifest['bytes']))
            # Publish a pointer only after the immutable generation is verified.
            run_prefix = prefix.rsplit('/archives/', 1)[0]
            _put_json(client, bucket, run_prefix + '/LATEST.json',
                      dict(destination=manifest['destination'], step=step,
                           manifest=prefix + '/archive_manifest.json'))
            _atomic_json(control / 'verified.json', manifest)
        except Exception as exc:
            _atomic_json(control / 'error.json', dict(stage='upload_or_verify', error=str(exc), time=time.time()))
            raise RuntimeError(f'实验归档失败，本地完整保留: {root}') from exc

        keep = bool(options.get('keep_local', False))
        if not keep:
            # Check the entire source again before deleting even an old checkpoint.
            if snapshot(root) != entries:
                raise RuntimeError('清理前本地实验发生修改；保留本地文件')
            try:
                for name, record in entries.items():
                    if not record['directory']:
                        p = root / name
                        s = p.lstat()
                        if (s.st_size, s.st_mtime_ns, s.st_ino, s.st_mode, s.st_dev) != (
                            record['size'], record['mtime_ns'], record['inode'], record['mode'], record['device']):
                            raise RuntimeError(f'清理时文件发生修改: {p}')
                        p.unlink()
                for name, rec in sorted(entries.items(), key=lambda x: len(Path(x[0]).parts), reverse=True):
                    if rec['directory']:
                        (root / name).rmdir()
                root.rmdir()
            except Exception as exc:
                _atomic_json(control / 'error.json', dict(stage='cleanup', error=str(exc),
                             destination=manifest['destination'], time=time.time()))
                raise RuntimeError(f'TOS 归档已验证，但本地清理未完成；检查 {control}') from exc
        result = dict(destination=manifest['destination'], step=step, files=len(selected),
                      bytes=manifest['bytes'], local_deleted=not keep, completed_at=time.time())
        _atomic_json(control / 'completed.json', result)
        logger.info('[archive] 完成: %s', result)
        return result


def finish_archive(output_path, config, accelerator, *, step, require_state):
    """One global uploader; other ranks poll shared storage, avoiding a long NCCL collective."""
    from omegaconf import OmegaConf
    from accelerate.utils import broadcast_object_list

    options = OmegaConf.to_container(config, resolve=True)
    if not options.get('enabled', True):
        return
    _close_run_logs(output_path)
    accelerator.wait_for_everyone()
    token = [uuid.uuid4().hex if accelerator.is_main_process else None]
    broadcast_object_list(token, from_process=0)
    status = Path(output_path).absolute().parent / '.tos_archives' / (Path(output_path).name + '.' + token[0] + '.json')
    if accelerator.is_main_process:
        try:
            result = archive_run(output_path, options, expected_step=step, require_state=require_state)
        except BaseException as exc:
            _atomic_json(status, dict(ok=False, error=f'{type(exc).__name__}: {exc}'))
            raise
        else:
            _atomic_json(status, dict(ok=True, result=result))
    else:
        deadline = time.monotonic() + float(options.get('wait_timeout_s', 86400))
        while not status.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f'等待主 rank 归档结果超时: {status}')
            time.sleep(1)
        result = json.loads(status.read_text())
        if not result['ok']:
            raise RuntimeError('主 rank 归档失败: ' + result['error'])

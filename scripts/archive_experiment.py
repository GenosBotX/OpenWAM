"""对已完成的 OpenWAM 实验重试归档。"""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwam.train.utils.experiment_archive import archive_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--root', help='覆盖实验配置中的 TOS 根路径')
    parser.add_argument('--path', help='root 下的相对路径，或完整 tos:// URI；默认沿用实验配置')
    parser.add_argument('--local-root')
    parser.add_argument('--credentials-file')
    parser.add_argument('--keep-local', action='store_true', default=None)
    args = parser.parse_args()
    # A weights-only run must explicitly have disabled resume state saving.
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(Path(args.run_dir) / 'config.yaml')
    require_state = bool(OmegaConf.select(cfg, 'training.save_full_states_for_resume', default=True))
    options = OmegaConf.to_container(OmegaConf.create(cfg.training.get('archive', {})), resolve=True)
    for key in ('root', 'path', 'local_root', 'credentials_file', 'keep_local'):
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    options.setdefault('credentials_file', os.environ.get(
        'OPENWAM_TOS_CREDENTIALS', '/vePFS/tangyuting/.config/openwam/tos-linglong.json'))
    result = archive_run(args.run_dir, options, require_state=require_state)
    print(result)


if __name__ == '__main__':
    main()

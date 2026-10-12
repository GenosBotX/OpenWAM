"""Create an evaluation-only Foundation bundle with the Linglong I/O contract.

No weights are changed/copied: the checkpoint and tokenizer are linked to the
original Foundation bundle. Local asset paths and the dataset normalization
contract are taken from the corresponding fine-tuned run for a controlled A/B.
"""
import argparse
import json
import shutil
from pathlib import Path

from omegaconf import OmegaConf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--foundation', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--arm-slots', action='store_true',
                   help='Map Linglong arm joints to the canonical left/right arm blocks.')
    args = p.parse_args()
    source, ref, out = (Path(v).resolve() for v in (args.foundation, args.reference, args.output))
    cfg = OmegaConf.load(source / 'config.yaml')
    reference = OmegaConf.load(ref / 'config.yaml')
    cfg.dataloader = reference.dataloader
    if args.arm_slots:
        cfg.dataloader.unify_action_map = ['0-6', 9, '34-40', 43, '68-75']
        cfg.dataloader.unify_state_map = None
    for key in ('model.video_backbone.model_path', 'model.video_backbone.encoder.model_path',
                'training.mixed_precision'):
        OmegaConf.update(cfg, key, OmegaConf.select(reference, key))
    # Structural parameters must match: local paths/freeze lists are not
    # structural and do not update any parameters at inference time.
    for key in ('model.architecture', 'model.action_backbone', 'model.video_backbone.components'):
        assert OmegaConf.to_container(OmegaConf.select(cfg, key)) == OmegaConf.to_container(OmegaConf.select(reference, key)), key
    out.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, out / 'config.yaml')
    shutil.copy2(ref / 'normalization_stats.npy', out / 'normalization_stats.npy')
    weights = source / 'checkpoint_step_154000.safetensors'
    (out / weights.name).symlink_to(weights)
    (out / 'tokenizer').symlink_to(source / 'tokenizer', target_is_directory=True)
    (out / 'evaluation_provenance.json').write_text(json.dumps({
        'weights': str(weights), 'reference_io_contract': str(ref),
        'unify_action_map': list(cfg.dataloader.unify_action_map),
        'note': 'All parameters are Foundation weights; only Linglong I/O metadata and local asset paths are adapted. Download SHA-256 previously failed official validation.',
    }, indent=2))
    print(out)


if __name__ == '__main__':
    main()

"""RTC contracts: sampling, real MoT gradients, masking and deploy re-referencing."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from openwam.model.rtc import DEFAULT_PREFIX_LENGTHS, sample_prefix_mask, suffix_padding_mask, validate_prefix_lengths


def _arch():
    from tests.test_architecture_variants import _build_arch
    arch = _build_arch('dual_system_self_attn', {
        'framework': 'dual_system', 'variant': 'joint_self_attn',
        'action_dim': 7, 'dim': 32, 'ffn_dim': 64, 'num_heads': 4,
        'text_dim': 64, 'bridge_layers': (0, 1),
    }, num_layers=2)
    arch.init_training_schedulers(1000)
    arch.configure_rtc({'enabled': True, 'prefix_lengths': list(DEFAULT_PREFIX_LENGTHS)}, action_horizon=32)
    return arch


def test_uniform_per_example_sampling():
    torch.manual_seed(12)
    mask = sample_prefix_mask(16000, 32, DEFAULT_PREFIX_LENGTHS, device='cpu')
    lengths, counts = mask.sum(1).unique(return_counts=True)
    assert lengths.tolist() == list(DEFAULT_PREFIX_LENGTHS)
    expected = 16000 / len(DEFAULT_PREFIX_LENGTHS)
    assert ((counts > expected * .9) & (counts < expected * 1.1)).all()
    assert not mask[:, 16:].any()


@pytest.mark.parametrize('lengths,horizon', [([],32), ([6,7],32), ([0,6,6],32), ([0,-1],32), ([0,True],32), ([0,6.0],32), ([0,12],12)])
def test_invalid_lengths(lengths, horizon):
    with pytest.raises(ValueError):
        validate_prefix_lengths(lengths, horizon)


@pytest.mark.parametrize('delay', DEFAULT_PREFIX_LENGTHS)
def test_mot_train_clean_prefix_and_backward(delay):
    from tests.test_openwam_trainer import _make_fake_loss_inputs
    arch = _arch().train()
    actions = torch.randn(2, 32, 7)
    prefix = torch.arange(32)[None, :].expand(2, -1) < torch.tensor([0, delay])[:, None]
    seen = []
    arch.register_forward_pre_hook(lambda _m, args: seen.append((args[0].detach().clone(), args[1].clone())))
    inputs = _make_fake_loss_inputs(B=2, T_action=32)
    inputs['context'] = torch.randn(2, 4, 64)
    inputs['action_is_pad'] = torch.zeros_like(actions, dtype=torch.bool)
    inputs['action_is_pad'][..., -2:] = True
    with patch('openwam.model.rtc.sample_prefix_mask', return_value=prefix):
        result = arch.compute_loss(**inputs, actions=actions)
    noisy, time = seen[0]
    assert time.shape == (2, 32)
    torch.testing.assert_close(noisy[prefix], actions[prefix], rtol=0, atol=0)
    assert (time[prefix] == 0).all()
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    grads = [p.grad for p in arch.action_backbone.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert arch.action_backbone.action_encoder.weight.grad.abs().sum() > 0


def test_suffix_loss_excludes_prefix_and_padding_and_normalizes_valid_cells():
    arch = _arch()
    prediction = torch.full((2, 32, 7), 3., requires_grad=True)
    actions = torch.zeros_like(prediction)
    prefix = torch.arange(32)[None, :] < torch.tensor([6,12])[:,None]
    padding = torch.zeros_like(actions, dtype=torch.bool)
    padding[..., -2:] = True
    mask = suffix_padding_mask(actions, padding, prefix)
    loss = arch._compute_action_loss(prediction, actions, torch.tensor([1,2]),
                                    SimpleNamespace(training_weight=lambda _: torch.ones(2)),
                                    {'action_is_pad': mask}, 'cpu')
    torch.testing.assert_close(loss, torch.tensor(9.))
    loss.backward()
    assert (prediction.grad[mask] == 0).all()
    assert (prediction.grad[~mask] != 0).all()


def test_invalid_training_windows():
    actions = torch.zeros(1,32,7)
    prefix = torch.arange(32)[None,:] < 6
    pad = torch.zeros_like(actions, dtype=torch.bool)
    pad[:,3] = True
    with pytest.raises(ValueError, match='prefix includes padded'):
        suffix_padding_mask(actions, pad, prefix)
    pad[:,3] = False
    pad[:,6:] = True
    with pytest.raises(ValueError, match='supervised suffix'):
        suffix_padding_mask(actions, pad, prefix)


def test_d0_per_token_time_equals_legacy_mot_time():
    from tests.test_wam_architecture import _dual_system_self_attn_mot_states
    arch = _arch().eval()
    vstate, astate = _dual_system_self_attn_mot_states(arch, 4)
    actions = astate.action_latents
    context = torch.randn(1,4,64)
    with torch.no_grad():
        legacy = arch.action_backbone.prepare_state(actions, torch.tensor([500.]), context=context)
        rtc = arch.action_backbone.prepare_state(actions, torch.full((1,3),500.), context=context)
        q1,k1,v1,post1 = arch.action_backbone.pre_attn_at_layer_for_compile(0,legacy)
        q2,k2,v2,post2 = arch.action_backbone.pre_attn_at_layer_for_compile(0,rtc)
    for a,b in zip((q1,k1,v1), (q2,k2,v2)):
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    for a,b in zip(post1[1:],post2[1:]):
        torch.testing.assert_close(a.expand_as(b),b,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('delay', DEFAULT_PREFIX_LENGTHS)
def test_generate_pins_prefix_every_step_and_preserves_inactive_noise(delay, monkeypatch):
    from tests.test_wam_architecture import _make_inactive_dim_probe_arch, _PIN_ACTIVE_IDX, _PIN_INACTIVE_IDX
    from openwam.deploy.optimizations.dit_cache import DiTVelocityCache
    monkeypatch.setattr(torch.compiler, 'cudagraph_mark_step_begin', lambda: None)
    arch = _make_inactive_dim_probe_arch()
    arch.rtc_prefix_lengths = DEFAULT_PREFIX_LENGTHS
    prefix = torch.zeros(delay,8)
    prefix[:,_PIN_ACTIVE_IDX] = 0.35
    times = []
    forward = arch.forward
    def observe(actions,time,**kw):
        times.append(time.clone())
        return forward(actions,time,**kw)
    arch.forward = observe
    mask = torch.zeros(8,dtype=torch.bool)
    mask[_PIN_ACTIVE_IDX] = True
    result = arch.generate(schedule=[(1000.,1000.),(500.,1000.),(0.,0.)],prompt='',
                           num_frames=33,action_prefix=prefix,active_action_mask=mask,
                           dit_cache=DiTVelocityCache(),decode_video=False)
    for seen,time in zip(arch.seen_action_inputs,times):
        torch.testing.assert_close(seen[0,:delay],prefix,atol=0,rtol=0)
        assert time.shape == (1,32)
        assert (time[:,:delay] == 0).all()
        assert (time[:,delay:] == 1000).all()
    np.testing.assert_array_equal(result['actions'][:delay],prefix.numpy())
    assert (result['actions'][:,_PIN_INACTIVE_IDX] == 0).all()


@pytest.mark.parametrize('delay', DEFAULT_PREFIX_LENGTHS)
def test_cached_rtc_skips_forwards_and_resets_between_changed_requests(delay, monkeypatch):
    from tests.test_wam_architecture import _make_inactive_dim_probe_arch, _PIN_ACTIVE_IDX, _PIN_INACTIVE_IDX
    from openwam.deploy.optimizations.dit_cache import DiTVelocityCache
    monkeypatch.setattr(torch.compiler, 'cudagraph_mark_step_begin', lambda: None)
    arch = _make_inactive_dim_probe_arch()
    arch.rtc_prefix_lengths = DEFAULT_PREFIX_LENGTHS
    cache = DiTVelocityCache()
    mask = torch.zeros(8, dtype=torch.bool)
    mask[_PIN_ACTIVE_IDX] = True
    kwargs = dict(schedule=[(1000.-i*200, 1000.-i*200) for i in range(6)],
                  prompt='', num_frames=33, decode_video=False, active_action_mask=mask)
    # Same cache object survives changed d, prefix values and random seeds.
    for d, value, seed in ((delay, .35, 7), (12 if delay != 12 else 6, -.25, 19)):
        prefix = torch.zeros(d, 8)
        prefix[:, _PIN_ACTIVE_IDX] = value
        arch.seen_action_inputs.clear()
        cached = arch.generate(**kwargs, action_prefix=prefix, seed=seed, dit_cache=cache)['actions']
        assert cache.stats['total_steps'] == 5
        assert cache.stats['total_skips'] == 3
        assert len(arch.seen_action_inputs) == 2
        for seen in arch.seen_action_inputs:
            torch.testing.assert_close(seen[0, :d], prefix, atol=0, rtol=0)
        np.testing.assert_array_equal(cached[:d], prefix.numpy())
        assert (cached[:, _PIN_INACTIVE_IDX] == 0).all()
        plain = arch.generate(**kwargs, action_prefix=prefix, seed=seed)['actions']
        # Constant-flow oracle: skipping must be exact here, not only close.
        np.testing.assert_array_equal(cached, plain)


def test_inference_rejects_old_checkpoint_and_untrained_length():
    from tests.test_wam_architecture import _make_inactive_dim_probe_arch
    arch = _make_inactive_dim_probe_arch()
    kw = dict(schedule=[],prompt='',num_frames=33,decode_video=False)
    with pytest.raises(ValueError,match='trained with RTC'):
        arch.generate(**kw,action_prefix=torch.zeros(6,8))
    arch.rtc_prefix_lengths = DEFAULT_PREFIX_LENGTHS
    with pytest.raises(ValueError,match='Unsupported RTC prefix length'):
        arch.generate(**kw,action_prefix=torch.zeros(5,8))


def _eef_state():
    return np.array([.2,.3,.4,1,0,0,0,1,0,.5, .6,.7,.8,1,0,0,0,1,0,.6],dtype=np.float32)


def test_wire_prefix_uses_new_reference_action_stats_and_action_scatter():
    from openwam.deploy.model_loader import _UnifyAwareNormalizer
    from openwam.dataloader.transforms.normalize import Normalizer
    from openwam.deploy.rtc import prepare_rtc_prefix
    from openwam.dataloader.utils.relative_eef import absolute_to_relative_eef20
    state = _eef_state()
    raw = np.repeat(state[None],6,axis=0)
    raw[:,0] += .1
    stats = {'min': np.full(20,-2.), 'max': np.full(20,2.),
             '_normalize_stats': {'min':np.full(20,-20.),'max':np.full(20,20.)}}
    inner = Normalizer('min_max',stats)
    action_map = np.r_[0:10,34:44]
    normalizer = _UnifyAwareNormalizer(inner,action_map,80,state_dst_index=np.arange(20)+50)
    arch = SimpleNamespace(rtc_prefix_lengths=DEFAULT_PREFIX_LENGTHS,normalizer=normalizer,action_dim=80)
    encoded, preserved = prepare_rtc_prefix(arch,raw,state)
    relative = absolute_to_relative_eef20(raw,state)
    np.testing.assert_allclose(encoded[:,action_map],relative/2,atol=1e-7)
    inactive = np.ones(80,dtype=bool); inactive[action_map] = False
    assert (encoded[:,inactive] == 0).all()
    np.testing.assert_array_equal(preserved,raw)
    normalized_state = normalizer.normalize(state)
    assert normalized_state[50] == pytest.approx(state[0]/20,abs=1e-7)
    empty,_ = prepare_rtc_prefix(arch,[],state)
    assert empty.shape == (0,80)


def test_rtc_config_inherits_current_training_setting(monkeypatch):
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    monkeypatch.setenv('OPENWAM_REPO_ROOT','/vePFS/liqi/OpenWAM-CUDAGraph')
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base=None):
        cfg = compose(config_name='train_piper_x_20261022_20261028_relative_eef_rtc_p16_16gpu_e5')
    assert tuple(cfg.training.rtc.prefix_lengths) == DEFAULT_PREFIX_LENGTHS
    assert cfg.training.rtc.enabled and cfg.dataloader.relative_to_state
    assert cfg.dataloader.num_frames == 33 and cfg.dataloader.video_stride == 4
    assert cfg.training.batch_size == 6 and cfg.training.gradient_accumulation_steps == 8
    assert cfg.training.num_epochs == 5 and cfg.training.archive.keep_local
    assert cfg.model.architecture.variant == 'joint_self_attn'
    assert '/liqi/' in cfg.training.output_path

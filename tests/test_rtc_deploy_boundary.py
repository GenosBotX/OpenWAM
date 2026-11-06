"""Exercise RTC through the real server, policy, engine and normalizer."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from openwam.dataloader.transforms.normalize import Normalizer
from openwam.dataloader.utils.relative_eef import absolute_to_relative_eef20
from openwam.deploy.engine import JointInferenceEngine
from openwam.deploy.model_loader import _UnifyAwareNormalizer, repr_contract_from_cfg
from openwam.deploy.server import PolicyServer
from openwam.model.rtc import DEFAULT_PREFIX_LENGTHS
from tests.test_piper_x_relative import config
from tests.test_relative_eef import pose


def _server(monkeypatch, *, rtc=True):
    monkeypatch.setattr('openwam.deploy.engine.make_schedule', lambda *a, **kw: object())
    cfg = config()
    stats = {'min': np.full(20, -2.), 'max': np.full(20, 2.),
             '_normalize_stats': {'min': np.full(20, -20.), 'max': np.full(20, 20.)}}
    action_map = np.r_[0:10, 34:44]
    normalizer = _UnifyAwareNormalizer(Normalizer('min_max', stats), action_map, 80)
    seen = []

    def generate(**kwargs):
        seen.append(kwargs)
        actions = np.tile(pose(.01, 0, 0, .2), (32, 1))
        encoded = kwargs.get('action_prefix')
        if encoded is not None and len(encoded):
            actions[:len(encoded)] = normalizer.unnormalize(encoded)
        return {'actions': actions, 'video': None}

    arch = SimpleNamespace(
        repr_contract=repr_contract_from_cfg(cfg), video_scheduler=None,
        action_scheduler=None, action_dim=80, normalizer=normalizer,
        rtc_prefix_lengths=DEFAULT_PREFIX_LENGTHS if rtc else (),
        normalize_deploy_proprio=normalizer.normalize, generate=generate,
    )
    return PolicyServer(JointInferenceEngine(cfg, architecture=arch), cfg), seen, normalizer


@pytest.mark.parametrize('delay', DEFAULT_PREFIX_LENGTHS)
def test_server_prefix_transport_references_raw_state_and_preserves_commands(monkeypatch, delay):
    server, seen, normalizer = _server(monkeypatch)
    state = pose(.4, .1, .2, -.3).astype(np.float32)
    prefix = np.repeat(state[None], delay, axis=0)
    # Deliberately outside training statistics to exercise clipping and exact
    # physical-prefix restoration at the engine boundary.
    prefix[:, 0] += 4 + np.arange(delay, dtype=np.float32) / 100
    response = server.predict({
        'request_id': delay, 'plan_version': 'queue-7', 'observation_mono_ns': 123456,
        'state': state.tolist(), 'prompt': 'pick', 'action_prefix': prefix.tolist(),
        'images': {'head_camera': Image.new('RGB', (32, 32))},
    })
    assert len(seen) == 1
    expected = normalizer.normalize_actions(absolute_to_relative_eef20(prefix, state))
    np.testing.assert_allclose(seen[0]['action_prefix'], expected, atol=1e-6)
    np.testing.assert_allclose(seen[0]['proprio'], normalizer.normalize(state), atol=1e-6)
    result = np.asarray(response['action'])
    assert result.shape == (32, 20)
    np.testing.assert_array_equal(result[:delay], prefix)
    assert result[delay, 0] == pytest.approx(.41, abs=1e-6)
    assert response['rtc_prefix_length'] == delay
    assert response['request_id'] == delay
    assert response['plan_version'] == 'queue-7'
    assert response['observation_mono_ns'] == 123456
    pong = server._pong_payload()
    assert pong['rtc_prefix_lengths'] == list(DEFAULT_PREFIX_LENGTHS)
    assert pong['rtc_prefix_format'] == 'absolute_eef20'
    assert pong['rtc_protocol_version'] == 1
    assert pong['rtc_prefix_anchor'] == 'observation_state'
    assert pong['rtc_prefix_index_origin'] == 'observation_action_zero'
    assert pong['rtc_conditioning'] == 'clean_prefix_zero_timestep'
    assert pong['num_actions'] == 32


@pytest.mark.parametrize('rtc,delay,message', [
    (False, 6, 'RTC-trained checkpoint'),
    (True, 5, 'Unsupported RTC prefix length'),
])
def test_server_rejects_unsupported_prefix_before_model_call(monkeypatch, rtc, delay, message):
    server, seen, _ = _server(monkeypatch, rtc=rtc)
    state = pose(.4, .1, .2, -.3)
    with pytest.raises(ValueError, match=message):
        server.predict({
            'state': state.tolist(), 'action_prefix': np.tile(state, (delay, 1)).tolist(),
            'images': {'head_camera': Image.new('RGB', (32, 32))},
        })
    assert not seen


def test_successive_requests_rebase_changed_prefix_against_own_state(monkeypatch):
    server, seen, normalizer = _server(monkeypatch)
    for rid, delay, x in [(10, 6, .4), (11, 12, .8), (12, 0, -.2), (13, 8, .1)]:
        state = pose(x, .1, .2, -.3).astype(np.float32)
        prefix = np.repeat(state[None], delay, axis=0)
        prefix[:, 0] += .01 * rid
        response = server.predict({
            'request_id': rid, 'plan_version': rid + 1, 'state': state.tolist(),
            'action_prefix': prefix.tolist(),
            'images': {'head_camera': Image.new('RGB', (32, 32))},
        })
        np.testing.assert_allclose(seen[-1]['action_prefix'],
            normalizer.normalize_actions(absolute_to_relative_eef20(prefix, state)), atol=1e-6)
        np.testing.assert_allclose(seen[-1]['proprio'], normalizer.normalize(state), atol=1e-6)
        np.testing.assert_array_equal(np.asarray(response['action'])[:delay], prefix)
        assert response['request_id'] == rid
        assert response['plan_version'] == rid + 1
        assert response['rtc_prefix_length'] == delay


@pytest.mark.parametrize('prefix', [None, [[1.], [1., 2.]], np.zeros((6, 19)).tolist(),
                                    np.full((6, 20), np.nan).tolist(), 'invalid'])
def test_bad_wire_prefix_is_validation_error_before_generation(monkeypatch, prefix):
    from openwam.deploy.obs_preprocess import ObsValidationError
    server, seen, _ = _server(monkeypatch)
    with pytest.raises(ObsValidationError, match='action_prefix'):
        server.predict({'state': pose(.4, .1, .2, -.3).tolist(), 'action_prefix': prefix,
            'images': {'head_camera': Image.new('RGB', (32, 32))}})
    assert not seen


@pytest.mark.parametrize('context', [{'request_id': []}, {'plan_version': True},
                                    {'observation_mono_ns': -1}, {'observation_mono_ns': 1.5}])
def test_bad_correlation_does_not_generate(monkeypatch, context):
    from openwam.deploy.obs_preprocess import ObsValidationError
    server, seen, _ = _server(monkeypatch)
    with pytest.raises(ObsValidationError):
        server.predict({**context, 'state': pose(.4, .1, .2, -.3).tolist(),
            'images': {'head_camera': Image.new('RGB', (32, 32))}})
    assert not seen


def test_legacy_request_does_not_require_or_invent_context(monkeypatch):
    server, seen, _ = _server(monkeypatch)
    response = server.predict({'state': pose(.4, .1, .2, -.3).tolist(),
        'images': {'head_camera': Image.new('RGB', (32, 32))}})
    assert len(seen) == 1
    assert not {'request_id', 'plan_version', 'observation_mono_ns', 'rtc_prefix_length'} & response.keys()

"""Request-local reconstruction before actions enter an execution buffer."""
import numpy as np

from openwam.dataloader.utils.relative_eef import (
    absolute_to_relative_eef20, relative_to_absolute_eef20, validate_relative_metadata,
)

RELATIVE_ACTION_OUTPUT_ABSOLUTE = "absolute"
RELATIVE_ACTION_OUTPUT_RELATIVE = "relative"


def relative_action_output_mode(cfg) -> str:
    """Return the wire-side relative-action mode from deploy config."""
    from omegaconf import OmegaConf

    mode = OmegaConf.select(cfg, "inference.relative_action_output", default=RELATIVE_ACTION_OUTPUT_ABSOLUTE)
    if mode is None:
        mode = RELATIVE_ACTION_OUTPUT_ABSOLUTE
    mode = str(mode).strip().lower()
    if mode not in (RELATIVE_ACTION_OUTPUT_ABSOLUTE, RELATIVE_ACTION_OUTPUT_RELATIVE):
        raise ValueError("inference.relative_action_output must be absolute or relative")
    return mode


def should_restore_relative_actions(cfg, contract) -> bool:
    """Whether the engine should reconstruct relative EEF20 into absolute poses."""
    contract = contract or {}
    relative_model = contract.get("model_action_target") == "relative_pose"
    mode = relative_action_output_mode(cfg)
    if mode == RELATIVE_ACTION_OUTPUT_RELATIVE and not relative_model:
        raise ValueError(
            "inference.relative_action_output=relative requires a relative-EEF checkpoint "
            "(model_action_target=relative_pose)"
        )
    return relative_model and mode == RELATIVE_ACTION_OUTPUT_ABSOLUTE


def apply_wire_relative_action_output(contract, cfg) -> dict:
    """Advertise the actions actually sent on the WebSocket, not the training default."""
    result = dict(contract or {})
    if relative_action_output_mode(cfg) != RELATIVE_ACTION_OUTPUT_RELATIVE:
        return result
    if result.get("model_action_target") != "relative_pose":
        raise ValueError(
            "inference.relative_action_output=relative requires a relative-EEF checkpoint "
            "(model_action_target=relative_pose)"
        )
    result["action_target"] = "relative_pose"
    result["relative_actions_decoded_by"] = "client"
    return result


def capture_action_reference(contract, raw_state):
    if contract.get("model_action_target") != "relative_pose":
        return None
    validate_relative_metadata(contract.get("model_action_contract", {}))
    if raw_state is None:
        raise ValueError("Relative EEF inference requires the observation's absolute EEF20 state")
    if hasattr(raw_state, "detach"):
        raw_state = raw_state.detach().float().cpu().numpy()
    state = np.array(raw_state, dtype=np.float32, copy=True)
    if state.shape == (1, 20):
        state = state[0].copy()
    if state.shape != (20,):
        raise ValueError("Relative EEF inference requires one raw absolute 20D state")
    absolute_to_relative_eef20(state, state)  # Validate both rotation frames before inference.
    return state


def restore_action_chunk(result, reference):
    if reference is None:
        return result
    actions = result["actions"]
    if hasattr(actions, "detach"):
        actions = actions.detach().float().cpu().numpy()
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 20 or len(actions) == 0:
        raise ValueError("Relative EEF inference must return a nonempty physical (T, 20) chunk")
    return {**result, "actions": relative_to_absolute_eef20(actions, reference)}

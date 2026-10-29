"""Request-local reconstruction before actions enter an execution buffer."""
import numpy as np

from openwam.dataloader.utils.relative_eef import (
    absolute_to_relative_eef20, relative_to_absolute_eef20, validate_relative_metadata,
)


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

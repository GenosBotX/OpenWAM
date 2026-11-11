"""Shared, auditable window selection for Piper-X readers and statistics."""
import json


def training_windows(storage, manifest, episodes, num_frames=33, stride=1):
    extent = 1 if manifest.get("static_hold", False) else num_frames
    selection = manifest.get("window_selection")
    if selection:
        if manifest.get("split") != "train" or stride != 1:
            raise ValueError("Explicit window selection requires train split and stride 1")
        payload = json.loads(storage.text(selection))
        if payload.get("format") != "piper_window_selection_v1":
            raise ValueError("Unsupported Piper-X window selection")
        windows = []
        for entry in payload["entries"]:
            episode, start, repeats = (entry[k] for k in ("episode_index", "start", "repeats"))
            if any(type(x) is not int for x in (episode, start, repeats)):
                raise ValueError("Window indices and repeats must be integers")
            if not (0 <= episode < len(episodes) and repeats > 0 and
                    0 <= start <= int(episodes[episode]["length"]) - extent):
                raise ValueError(f"Invalid selected window: {entry}")
            windows.extend([(episode, start)] * repeats)
        if len(windows) != manifest["total_windows"]:
            raise ValueError("Selected windows do not match manifest total_windows")
        return windows
    return [(i, start) for i, episode in enumerate(episodes)
            for start in range(0, max(0, int(episode["length"]) - extent + 1), stride)]

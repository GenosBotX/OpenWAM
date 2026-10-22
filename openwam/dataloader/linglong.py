"""Linglong decoded / OpenWAM parquet reader.

The reader uses matching joint/joint or EEF/EEF state and action columns,
normalizes each stream with its own training statistics,
then optionally scatters both into OpenWAM's 80-D unified space.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.tos_cache import BoundedTosCache, DatasetStorage
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.dataloader.utils import get_cfg
from openwam.dataloader.utils.normalization import apply_normalization
from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, parse_unify_spec


ACTION_DIM = 24
STATE_DIM = 24
NUM_FRAMES = 33
VIDEO_STRIDE = 4
VIDEO_INDICES = np.arange(0, NUM_FRAMES, VIDEO_STRIDE, dtype=np.int64)
RAW_DIM_MASK = np.zeros(ACTION_DIM, dtype=bool)
RAW_DIM_MASK[:22] = True  # base[22:24] are documented placeholders in Linglong.


class LinglongDataset(BaseDataset):
    """Reader for ``linglong_0923_openwam_{train,test}_15hz`` buckets."""

    @classmethod
    def from_config(cls, config, split: str = "train"):
        raw_dir = os.environ.get("OPENWAM_LINGLONG_DATASET_DIR") or str(get_cfg(config, "dataset_dir"))
        remote = raw_dir.startswith("tos://")
        dataset_dir = raw_dir if remote else Path(raw_dir)
        if not remote and not dataset_dir.is_dir():
            dataset_dir = cls._resolve_dataset_dir(dataset_dir)
        # Configs point at the training bucket; select the sibling test bucket
        # automatically for evaluation/validation builds.
        if remote and split in ("test", "val"):
            dataset_dir = get_cfg(config, "eval_dataset_dir", None)
            if not dataset_dir:
                raise ValueError("TOS evaluation requires explicit eval_dataset_dir")
        if not remote and split in ("test", "val") and dataset_dir.name.endswith("_train_15hz"):
            candidate = dataset_dir.with_name(dataset_dir.name.replace("_train_15hz", "_test_15hz"))
            if candidate.is_dir():
                dataset_dir = candidate
        return cls(
            dataset_dir=str(dataset_dir),
            split=split,
            num_frames=int(get_cfg(config, "num_frames", NUM_FRAMES)),
            video_stride=int(get_cfg(config, "video_stride", VIDEO_STRIDE)),
            window_stride=int(get_cfg(config, "window_stride", 1)),
            height=int(get_cfg(config, "height", 384)),
            width=int(get_cfg(config, "width", 320)),
            normalize_mode=get_cfg(config, "normalize_mode", "min-max"),
            camera_layout=list(get_cfg(config, "camera_layout", ["head", "left", "right"])),
            multiview=bool(get_cfg(config, "multiview", True)),
            overfit_single=bool(get_cfg(config, "overfit_single", False)),
            overfit_episode=int(get_cfg(config, "overfit_episode", 0)),
            overfit_repeats=int(get_cfg(config, "overfit_repeats", 1)),
            unify_action=bool(get_cfg(config, "unify_action", True)),
            unify_action_map=get_cfg(config, "unify_action_map", None),
            unify_state_map=get_cfg(config, "unify_state_map", None),
            action_key=get_cfg(config, "action_key", "action"),
            state_key=get_cfg(config, "state_key", "observation.state"),
            action_mode=get_cfg(config, "action_mode", "joint"),
            normalization_dataset_dir=get_cfg(config, "normalization_dataset_dir", None),
            active_raw_action_dims=get_cfg(config, "active_raw_action_dims", None),
            active_raw_state_dims=get_cfg(config, "active_raw_state_dims", None),
            cache_dir=get_cfg(config, "cache_dir", "/tmp/openwam-tos-cache"),
            cache_max_bytes=int(get_cfg(config, "cache_max_bytes", 4 * 1024**3)),
            cache_min_free_bytes=int(get_cfg(config, "cache_min_free_bytes", 1024**3)),
        )

    def __init__(
        self,
        dataset_dir: str,
        *,
        split: str = "train",
        num_frames: int = NUM_FRAMES,
        video_stride: int = VIDEO_STRIDE,
        window_stride: int = 1,
        height: int = 384,
        width: int = 320,
        normalize_mode: Optional[str] = "min-max",
        camera_layout: Optional[list[str]] = None,
        multiview: bool = True,
        overfit_single: bool = False,
        overfit_episode: int = 0,
        overfit_repeats: int = 1,
        unify_action: bool = True,
        unify_action_map=None,
        unify_state_map=None,
        action_key: str = "action",
        state_key: str = "observation.state",
        action_mode: str = "joint",
        normalization_dataset_dir=None,
        active_raw_action_dims=None,
        active_raw_state_dims=None,
        cache_dir="/tmp/openwam-tos-cache",
        cache_max_bytes=4 * 1024**3,
        cache_min_free_bytes=1024**3,
    ):
        dataset_dir = str(dataset_dir)
        self._remote = dataset_dir.startswith("tos://")
        self.dataset_dir = dataset_dir if self._remote else self._resolve_dataset_dir(dataset_dir)
        needs_cache = self._remote or str(normalization_dataset_dir).startswith("tos://")
        self.tos_cache = BoundedTosCache(cache_dir, cache_max_bytes, cache_min_free_bytes) if needs_cache else None
        self._storage = DatasetStorage(self.dataset_dir, self.tos_cache)
        self.split = str(split)
        self.action_key = str(action_key)
        self.state_key = str(state_key)
        self.action_mode = str(action_mode)
        if (self.action_key, self.action_mode) not in (("action", "joint"), ("action_eef", "eef_quat")):
            raise ValueError("Linglong requires action/joint or action_eef/eef_quat")
        self.raw_action_dim = 16 if self.action_key == "action_eef" else ACTION_DIM
        if self.state_key not in ("observation.state", "observation.state_eef"):
            raise ValueError("Linglong state_key must be observation.state or observation.state_eef")
        if self.state_key == "observation.state_eef" and self.action_key != "action_eef":
            raise ValueError("EEF state requires action_eef targets")
        if self.action_key == "action_eef" and self.state_key != "observation.state_eef":
            raise ValueError("Linglong EEF actions require observation.state_eef; joint-state/EEF-action experiments are retired")
        self.raw_state_dim = 16 if self.state_key == "observation.state_eef" else STATE_DIM
        self.raw_action_mask = self._raw_mask(
            active_raw_action_dims, self.raw_action_dim,
            16 if self.action_key == "action_eef" else 22,
        )
        self.raw_state_mask = self._raw_mask(
            active_raw_state_dims, self.raw_state_dim, 16 if self.raw_state_dim == 16 else 22,
        )
        self.num_frames = int(num_frames)
        self.video_stride = int(video_stride)
        self.window_stride = int(window_stride)
        self.height = int(height)
        self.width = int(width)
        self.normalize_mode = None if normalize_mode in (None, "", "none", "null") else str(normalize_mode)
        self.camera_layout = camera_layout or ["head", "left", "right"]
        self.multiview = bool(multiview)
        self.overfit_single = bool(overfit_single)
        self.overfit_episode = int(overfit_episode)
        self.overfit_repeats = int(overfit_repeats)
        if self.overfit_single and self.overfit_repeats < 1:
            raise ValueError("Linglong overfit_repeats must be >= 1")
        self.unify_action = bool(unify_action)
        if self.unify_action:
            if unify_action_map is None:
                raise ValueError("Linglong unify_action=true requires an explicit unify_action_map")
            self._action_dst_index = parse_unify_spec(unify_action_map, UNIFY_DIM)
            state_spec = unify_state_map if unify_state_map is not None else unify_action_map
            self._state_dst_index = parse_unify_spec(state_spec, UNIFY_DIM)
            if self._action_dst_index.shape != (self.raw_action_dim,) or self._state_dst_index.shape != (self.raw_state_dim,):
                raise ValueError(f"Linglong maps must cover action={self.raw_action_dim}, state={self.raw_state_dim} dimensions")
            self._action_dim = UNIFY_DIM
            self._state_dim = UNIFY_DIM
        else:
            self._action_dst_index = None
            self._state_dst_index = None
            self._action_dim = self.raw_action_dim
            self._state_dim = self.raw_state_dim
        if self.num_frames < 2 or self.video_stride < 1:
            raise ValueError("num_frames must be >=2 and video_stride must be >=1")
        if self.num_frames != NUM_FRAMES or self.video_stride != VIDEO_STRIDE:
            raise ValueError("Linglong converted buckets require num_frames=33 and video_stride=4")
        if self.multiview and len(self.camera_layout) != 3:
            raise ValueError("Linglong multiview camera_layout must contain head, left, right")

        manifest_path = f"{self.dataset_dir}/manifest.json"
        if not self._remote and not Path(manifest_path).is_file():
            raise FileNotFoundError(
                "Linglong manifest is missing: "
                f"{manifest_path}. Set dataloader.dataset_dir (or "
                "OPENWAM_LINGLONG_DATASET_DIR) to the mounted converted bucket."
            )
        manifest = json.loads(self._storage.text("manifest.json"))
        if self._remote and manifest.get("status") != "complete":
            raise ValueError("TOS training requires a complete published dataset")
        self._eef_valid_column = manifest.get("eef_valid_column")
        if self._eef_valid_column is not None:
            if (self._eef_valid_column != "eef_valid" or self.raw_action_dim != 16
                    or self.raw_state_dim != 16 or self.action_mode != "eef_quat"
                    or manifest.get("eef_valid_layout") != ["left", "right"]):
                raise ValueError("Per-hand EEF validity requires left/right 16D EEF state and actions")
        if self.state_key == "observation.state_eef":
            state_contract = manifest.get("observation_state_eef", {})
            if (state_contract.get("column") != self.state_key or state_contract.get("shape") != [16]
                    or state_contract.get("reference") != "base_link"
                    or state_contract.get("quaternion_order") != "wxyz"):
                raise ValueError("EEF state requires a converted bucket with its 16D base_link/wxyz contract")
        if self.action_key == "action_eef" and self.split in ("test", "val") and manifest.get("split") == "train":
            raise ValueError("EEF test/val requires a separate held-out EEF bucket, not the training bucket")
        image_root = manifest.get("source_root")
        if self._remote:
            # Published portable buckets keep images under their own immutable prefix.
            if image_root not in (None, "."):
                raise ValueError("TOS Linglong manifest source_root must be '.'; publish a portable bucket first")
            self._source_root = self.dataset_dir
        elif image_root is not None and not Path(image_root).is_absolute():
            image_root = str(self.dataset_dir / image_root)
        if not self._remote:
            self._source_root = self._resolve_source_root(image_root)
        self._images = DatasetStorage(self._source_root, self.tos_cache)
        self._prompt = str(manifest["prompt"])
        if not self._remote and not self._source_root.is_dir():
            raise FileNotFoundError(
                "Linglong source_root is unavailable: "
                f"{self._source_root}. Set OPENWAM_LINGLONG_SOURCE_ROOT to the "
                "shared decoded image root on the training nodes."
            )
        self._episodes = [json.loads(line) for line in self._storage.text("meta/episodes.jsonl").splitlines() if line.strip()]
        self._tables = [
            f"data/chunk-{i // 1000:03d}/episode_{i:06d}.parquet"
            for i in range(len(self._episodes))
        ]
        self._windows = []
        for local, episode in enumerate(self._episodes):
            length = int(episode["length"])
            max_start = length - self.num_frames if self.split in ("val", "test") else length - self.num_frames
            if max_start >= 0:
                self._windows.extend((local, start) for start in range(0, max_start + 1, self.window_stride))
        if not self._windows:
            raise ValueError(f"Linglong split={self.split!r} contains no full {self.num_frames}-frame windows")
        self._base_num_windows = len(self._windows)
        if self.overfit_single:
            if self.overfit_episode < 0 or self.overfit_episode >= len(self._episodes):
                raise ValueError(
                    f"Linglong overfit_episode={self.overfit_episode} is outside "
                    f"the available episode range [0, {len(self._episodes) - 1}]"
                )
            # "single sample" means one complete episode: retain every valid
            # sliding window from that episode, rather than repeating one
            # 33-frame window. Optional repeats duplicate the episode window
            # sequence for deliberate overfit runs.
            episode_windows = [item for item in self._windows if item[0] == self.overfit_episode]
            if not episode_windows:
                raise ValueError(
                    f"Linglong overfit_episode={self.overfit_episode} has no full "
                    f"{self.num_frames}-frame windows"
                )
            self._windows = episode_windows * self.overfit_repeats

        stats_dir = self.dataset_dir
        if normalization_dataset_dir is not None:
            # Stats paths must not follow the observation-bucket environment override.
            stats_dir = str(normalization_dataset_dir)
            if not stats_dir.startswith("tos://"):
                stats_dir = self._resolve_dataset_dir(stats_dir, use_env=False)
        elif self.action_key == "action_eef" and self.split in ("test", "val"):
            raise ValueError("EEF evaluation requires normalization_dataset_dir pointing at the training bucket")
        stats_storage = DatasetStorage(stats_dir, self.tos_cache)
        stats = json.loads(stats_storage.text("meta/normalization_stats.json"))
        self._state_stats = {k: np.asarray(v, dtype=np.float32) for k, v in stats["state"].items() if k != "count"}
        self._action_stats = {k: np.asarray(v, dtype=np.float32) for k, v in stats["action"].items() if k != "count"}
        if self.action_key == "action_eef":
            if json.loads(stats_storage.text("manifest.json")).get("split") != "train":
                raise ValueError("EEF normalization statistics must come from the training bucket")
            eef_stats = json.loads(stats_storage.text("meta/action_eef_stats.json"))
            if eef_stats.get("source_column") != "action_eef":
                raise ValueError("EEF statistics must describe action_eef")
            self._action_stats = {
                k: np.asarray(eef_stats["action"]["default"][f"global_{k}"], dtype=np.float32)
                for k in ("mean", "std", "min", "max", "q01", "q99")
            }
            if self.state_key == "observation.state_eef":
                state_eef_stats = json.loads(stats_storage.text("meta/state_eef_stats.json"))
                if state_eef_stats.get("source_column") != self.state_key:
                    raise ValueError("EEF state statistics must describe observation.state_eef")
                self._state_stats = {
                    k: np.asarray(state_eef_stats["state"]["default"][f"global_{k}"], dtype=np.float32)
                    for k in ("mean", "std", "min", "max", "q01", "q99")
                }
        for label, values, dim in (("action", self._action_stats, self.raw_action_dim),
                                    ("state", self._state_stats, self.raw_state_dim)):
            for key in ("mean", "std", "min", "max", "q01", "q99"):
                if values[key].shape != (dim,) or not np.isfinite(values[key]).all():
                    raise ValueError(f"Invalid {label} normalization statistic {key}, expected {dim} finite values")
        if self.normalize_mode not in (None, "min-max", "z-score", "quantile"):
            raise ValueError(f"unsupported Linglong normalize_mode={self.normalize_mode!r}")

        # Training copies this artifact into the checkpoint for deployment.
        # Keep action and proprio statistics separate for deployment.
        if self.action_key == "action_eef":
            # Export the exact transforms used above directly into the run.
            # Never overwrite the joint statistics in the source data copy.
            self.normalization_stats_payload = {
                "eef_quat": self._action_stats,
                "eef_quat_state": self._state_stats,
            }
            self.normalization_stats_path = None
            return
        if stats_storage.remote:
            # Joint deployment artifact must remain available after cache eviction.
            with stats_storage.open("meta/normalization_stats.npy") as artifact:
                self.normalization_stats_payload = np.load(artifact, allow_pickle=True).item()
            self.normalization_stats_path = None
            return
        stats_artifact = Path(stats_dir) / "meta" / "normalization_stats.npy"
        if not stats_artifact.is_file():
            raise FileNotFoundError(f"Linglong deploy normalization artifact is missing: {stats_artifact}")
        self.normalization_stats_path = str(stats_artifact)

    @staticmethod
    def _raw_mask(dims, width, default_count):
        indices = np.asarray(list(range(default_count)) if dims is None else list(dims), dtype=np.int64)
        if (not len(indices) or len(np.unique(indices)) != len(indices)
                or indices.min() < 0 or indices.max() >= width):
            raise ValueError(f"Active raw dimensions must be unique indices within [0, {width})")
        mask = np.zeros(width, dtype=bool)
        mask[indices] = True
        return mask

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @staticmethod
    def _resolve_dataset_dir(dataset_dir, *, use_env=True) -> Path:
        """Resolve the converted bucket across local/Volcano mount aliases.

        The launcher may pass a path using the vePFS prefix while a worker
        exposes the same tree through the development prefix (or vice versa).
        Probe both forms before opening ``manifest.json``.  An explicit
        environment override is useful for clusters with a site-specific
        mount point.
        """
        raw = str(dataset_dir)
        candidates = []
        override = os.environ.get("OPENWAM_LINGLONG_DATASET_DIR") if use_env else None
        if override:
            candidates.append(Path(override))
        candidates.append(Path(raw))

        if raw.startswith("/iag_ad_01/ad/"):
            suffix = raw[len("/iag_ad_01/ad/") :]
            candidates.extend(
                [
                    Path("/iag_ad_vepfs_volc/iag_ad_vepfs_volc") / suffix,
                    Path("/iag_ad_vepfs_volc") / suffix,
                ]
            )
        if raw.startswith("/iag_ad_vepfs_volc/iag_ad_vepfs_volc/"):
            suffix = raw[len("/iag_ad_vepfs_volc/iag_ad_vepfs_volc/") :]
            candidates.extend(
                [
                    Path("/iag_ad_01/ad") / suffix,
                    Path("/iag_ad_vepfs_volc") / suffix,
                ]
            )
        elif raw.startswith("/iag_ad_vepfs_volc/"):
            suffix = raw[len("/iag_ad_vepfs_volc/") :]
            candidates.extend(
                [
                    Path("/iag_ad_01/ad") / suffix,
                    Path("/iag_ad_vepfs_volc/iag_ad_vepfs_volc") / suffix,
                ]
            )

        seen = set()
        unique = []
        for candidate in candidates:
            candidate = candidate.expanduser()
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
            if (candidate / "manifest.json").is_file():
                return candidate

        raise FileNotFoundError(
            "Linglong dataset bucket/manifest.json not found. Tried: "
            + ", ".join(str(path / "manifest.json") for path in unique)
        )

    @staticmethod
    def _resolve_source_root(manifest_root) -> Path:
        """Resolve image root across local and Volcano vePFS mount aliases.

        The converter records the path visible on the machine where it runs.
        Volcano workers mount the same vePFS tree under a different prefix, so
        the manifest path may not exist even though the data is present.
        ``OPENWAM_LINGLONG_SOURCE_ROOT`` is an explicit override; otherwise
        try the known vePFS aliases before returning the original path.
        """
        override = os.environ.get("OPENWAM_LINGLONG_SOURCE_ROOT")
        candidates = []
        if override:
            candidates.append(Path(override))
        if manifest_root:
            raw = str(manifest_root)
            candidates.append(Path(raw))
            # /iag_ad_01/ad/... on the dev host -> shared vePFS mount on
            # Volcano workers.
            if raw.startswith("/iag_ad_01/ad/"):
                candidates.append(Path("/iag_ad_vepfs_volc/iag_ad_vepfs_volc/") / raw[len("/iag_ad_01/ad/") :])
            if raw.startswith("/iag_ad_vepfs_volc/"):
                candidates.append(Path("/iag_ad_vepfs_volc/iag_ad_vepfs_volc/") / raw[len("/iag_ad_vepfs_volc/") :])
                if raw.startswith("/iag_ad_vepfs_volc/iag_ad_vepfs_volc/"):
                    candidates.append(Path("/iag_ad_01/ad/") / raw[len("/iag_ad_vepfs_volc/iag_ad_vepfs_volc/") :])
            # Be tolerant of a path serialized without its leading '/'.
            if not raw.startswith("/"):
                candidates.append(Path("/iag_ad_vepfs_volc/iag_ad_vepfs_volc") / raw)
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return candidates[0] if candidates else Path(".")

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def normalization_stats(self) -> dict:
        return {"action": dict(self._action_stats), "state": dict(self._state_stats)}

    def __len__(self) -> int:
        return len(self._windows)

    @lru_cache(maxsize=128)
    def _read_table(self, local: int):
        with self._storage.open(self._tables[local]) as handle:
            return pq.read_table(
                handle,
                columns=[
                    self.state_key,
                    self.action_key,
                    "observation.images.head",
                    "observation.images.left",
                    "observation.images.right",
                ] + ([self._eef_valid_column] if self._eef_valid_column else []),
            )

    def _norm(self, values: np.ndarray, stats: dict) -> np.ndarray:
        return np.asarray(apply_normalization(values, stats, self.normalize_mode), dtype=np.float32)

    def _image(self, rel: str) -> Image.Image:
        with self._images.open(rel) as handle:
            with Image.open(handle) as image:
                return image.convert("RGB")

    def __getitem__(self, index: int) -> dict:
        local, start = self._windows[index]
        table = self._read_table(local).slice(start, self.num_frames)
        state = np.asarray(table[self.state_key].to_pylist(), dtype=np.float32)
        action = np.asarray(table[self.action_key].to_pylist(), dtype=np.float32)
        if state.shape != (self.num_frames, self.raw_state_dim) or action.shape != (self.num_frames, self.raw_action_dim):
            raise ValueError(f"Linglong row shape mismatch: state={state.shape}, action={action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("Non-finite Linglong action/state")
        if self.action_key == "action_eef":
            for start_q in (3, 11):
                if not np.allclose(np.linalg.norm(action[:, start_q:start_q + 4], axis=-1), 1, atol=1e-3):
                    raise ValueError("action_eef must contain unit wxyz quaternions")
        if self.state_key == "observation.state_eef":
            for start_q in (3, 11):
                if not np.allclose(np.linalg.norm(state[:, start_q:start_q + 4], axis=-1), 1, atol=1e-3):
                    raise ValueError("observation.state_eef must contain unit wxyz quaternions")

        frames = []
        for t in VIDEO_INDICES:
            cams = {
                "head": self._image(table["observation.images.head"][int(t)].as_py()),
                "left": self._image(table["observation.images.left"][int(t)].as_py()),
                "right": self._image(table["observation.images.right"][int(t)].as_py()),
            }
            if self.multiview:
                frames.append(assemble_multiview_layout(cams, self.camera_layout, self.height, self.width))
            else:
                frames.append(cams["head"].resize((self.width, self.height), Image.BILINEAR))

        # OpenWAM-α always consumes an 80-D unified vector.  Normalize in the
        # selected raw action/state spaces first, then scatter into the configured
        # slots; this is the inverse of deployment's gather-then-unnormalize.
        raw_action = self._norm(action[: self.num_frames - 1], self._action_stats)
        raw_proprio = self._norm(state[:1], self._state_stats)
        # Absent dimensions and invalid hand labels must not enter the action
        # or proprio encoders, including after normalization.
        action_valid = np.broadcast_to(self.raw_action_mask, raw_action.shape).copy()
        proprio_valid = np.broadcast_to(self.raw_state_mask, raw_proprio.shape).copy()
        if self._eef_valid_column:
            validity = np.asarray(table[self._eef_valid_column].to_pylist())
            if validity.shape != (self.num_frames, 2) or validity.dtype != np.bool_:
                raise ValueError("eef_valid must contain boolean left/right flags per row")
            if not validity.any(axis=1).all():
                raise ValueError("Masked EEF segment contains a row with no valid hand")
            action_valid &= np.repeat(validity[:self.num_frames - 1], 8, axis=1)
            proprio_valid &= np.repeat(validity[:1], 8, axis=1)
        raw_action[~action_valid] = 0.0
        raw_proprio[~proprio_valid] = 0.0
        if self.unify_action:
            action_out, _ = map_to_unify(raw_action, self._action_dst_index, UNIFY_DIM)
            proprio_out, _ = map_to_unify(raw_proprio, self._state_dst_index, UNIFY_DIM)
            action_valid_out = np.zeros(action_out.shape, dtype=bool)
            proprio_valid_out = np.zeros(proprio_out.shape, dtype=bool)
            action_valid_out[:, self._action_dst_index] = action_valid
            proprio_valid_out[:, self._state_dst_index] = proprio_valid
        else:
            action_out, proprio_out = raw_action, raw_proprio
            action_valid_out, proprio_valid_out = action_valid, proprio_valid
        action_mask = torch.from_numpy(action_valid_out)
        proprio_mask = torch.from_numpy(proprio_valid_out)
        return {
            "video": frames,
            "vace_video": None,
            "first_frame_image": [frames[0]],
            "action": torch.from_numpy(action_out),
            "action_mask": action_mask,
            "video_mask": torch.ones(len(frames), dtype=torch.bool),
            "proprio": torch.from_numpy(proprio_out),
            "proprio_mask": proprio_mask,
            "prompt": (self._episodes[local].get("tasks") or [self._prompt])[0],
            "episode_index": int(self._episodes[local]["episode_index"]),
            "start_frame": int(start),
        }

    @staticmethod
    def _scatter_mask(dst_index: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
        mask = np.zeros(UNIFY_DIM, dtype=bool)
        mask[dst_index[raw_mask]] = True
        return mask
__all__ = ["LinglongDataset"]

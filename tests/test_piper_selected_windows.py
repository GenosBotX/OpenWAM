"""Exercise selected-window weights and mounted references through the reader."""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from openwam.dataloader.piper_x import PiperXDataset, STATE_COLUMN, ACTION_COLUMN
from openwam.dataloader.tos_cache import DatasetStorage
from tests.test_piper_x_static import make_bucket, reader
from tests.test_piper_x_relative import stats_file


def selected(root, entries):
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.update(window_selection="windows.json", total_windows=sum(x["repeats"] for x in entries))
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "windows.json").write_text(json.dumps({"format": "piper_window_selection_v1", "entries": entries}))


def test_selected_transition_hold_and_repeats(tmp_path):
    root = make_bucket(tmp_path / "train", False, [40])
    path = root / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(path)
    tail = table.slice(39, 1).take(pa.array([0] * 33))
    tail = tail.set_column(tail.schema.get_field_index(ACTION_COLUMN), ACTION_COLUMN, tail[STATE_COLUMN])
    pq.write_table(pa.concat_tables([table, tail]), path)
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 73}) + "\n")
    selected(root, [dict(episode_index=0, start=30, repeats=3), dict(episode_index=0, start=40, repeats=1)])
    ds = reader(root)
    assert len(ds) == 4 and ds._windows == [(0, 30)] * 3 + [(0, 40)]
    transition, hold = ds[0], ds[3]
    assert transition["action"][0, 0] != transition["action"][-1, 0]
    assert torch.equal(ds[0]["action"], ds[2]["action"])
    assert torch.equal(hold["action"], hold["proprio"].expand(32, -1))
    assert all(np.array_equal(hold["video"][0], image) for image in hold["video"])


@pytest.mark.parametrize("entry", [dict(episode_index=0, start=2, repeats=1),
                                      dict(episode_index=0, start=0, repeats=0),
                                      dict(episode_index=1, start=0, repeats=1)])
def test_invalid_selection_is_rejected(tmp_path, entry):
    root = make_bucket(tmp_path / "train", False, [34])
    selected(root, [entry])
    with pytest.raises(ValueError, match="Invalid selected window"):
        reader(root)


def test_selection_total_mismatch_is_rejected(tmp_path):
    root = make_bucket(tmp_path / "train", False, [34])
    selected(root, [dict(episode_index=0, start=0, repeats=2)])
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["total_windows"] = 1
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="total_windows"):
        reader(root)


@pytest.mark.parametrize("stale", [False, True])
def test_tos_reader_resolves_mounted_images_and_checks_stats(tmp_path, monkeypatch, stale):
    root = make_bucket(tmp_path / "train", False, [34])
    selected(root, [dict(episode_index=0, start=1, repeats=3)])
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["source_root"] = "/tos/piperx/openwam/original/train"
    (root / "manifest.json").write_text(json.dumps(manifest))
    payload = stats_file(root)
    payload["sources"] = {"/tos/piperx/openwam/halfhold/train": {
        "windows": 3,
        "window_selection_sha256": "stale" if stale else hashlib.sha256((root / "windows.json").read_bytes()).hexdigest(),
    }}
    (root / "meta/relative_eef20_normalization_stats.json").write_text(json.dumps(payload))

    def local(storage, relative):
        assert str(storage.root) in {
            "tos://test/piperx/openwam/halfhold/train", "tos://test/piperx/openwam/original/train",
        }
        return root / relative

    monkeypatch.setattr(DatasetStorage, "text", lambda self, rel: local(self, rel).read_text())

    @contextmanager
    def open_local(self, rel):
        with local(self, rel).open("rb") as handle:
            yield handle

    monkeypatch.setattr(DatasetStorage, "open", open_local)
    kwargs = dict(action_mode="eef", relative_to_state=True, height=32, width=32, cache_dir=str(tmp_path / "cache"))
    if stale:
        with pytest.raises(ValueError, match="do not match selected windows"):
            PiperXDataset("tos://test/piperx/openwam/halfhold/train", **kwargs)
    else:
        ds = PiperXDataset("tos://test/piperx/openwam/halfhold/train", **kwargs)
        assert ds._source_root == "tos://test/piperx/openwam/original/train"
        assert len(ds) == 3 and ds[0]["start_frame"] == 1
        assert ds[0]["action"].shape == (32, 80) and len(ds[0]["video"]) == 9

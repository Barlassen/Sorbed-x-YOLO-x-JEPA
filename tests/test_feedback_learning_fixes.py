"""Council-driven correctness fixes to the continual + federated learning path.

* continual: only independent human corrections are replay-eligible (a bare
  confirmation is not); grader_sha provenance survives the manifest round-trip.
* federated: LoRA factors are never factor-averaged; client updates must share a
  base checkpoint and round id; per-client delta norm can be clipped.
"""

from __future__ import annotations

import pytest
from training.continual import (
    ContinualRow,
    append_replay_rows,
    read_continual_manifest,
)


def _row(source: str, grader_sha: str = "abc123", tmp=None) -> ContinualRow:
    return ContinualRow(
        image=(tmp / "img.png") if tmp else __import__("pathlib").Path("img.png"),
        mask=None, stage_index=2, patient_id="p1", sample_weight=1.0,
        source=source, grader_sha=grader_sha,
    )


def test_confirmation_is_not_replay_eligible() -> None:
    assert _row("human_corrected").is_human is True
    assert _row("correction").is_human is True
    # A bare agree must NOT seed the replay buffer (closes the echo chamber).
    assert _row("human_confirmed").is_human is False
    assert _row("rule").is_human is False


def test_grader_sha_survives_replay_round_trip(tmp_path) -> None:
    replay = tmp_path / "replay.csv"
    rows = [_row("human_corrected", grader_sha="sha_of_grader", tmp=tmp_path)]
    (tmp_path / "img.png").write_bytes(b"x")
    append_replay_rows(replay, rows)
    reloaded = read_continual_manifest(replay)
    assert reloaded[0].grader_sha == "sha_of_grader"
    assert reloaded[0].source == "human_corrected"


def test_manifest_without_grader_sha_still_loads(tmp_path) -> None:
    # Backward compatibility: the six original columns are enough to read.
    path = tmp_path / "legacy.csv"
    (tmp_path / "i.png").write_bytes(b"x")
    path.write_text(
        "image,mask,stage,patient_id,sample_weight,source\n"
        f"{tmp_path / 'i.png'},,stage_3,p1,1.0,rule\n",
        encoding="utf-8",
    )
    rows = read_continual_manifest(path)
    assert rows[0].grader_sha == ""  # absent -> empty, not a crash


# --------------------------------------------------------------------------- #
# Federated aggregation
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")

from training.fl_client import (  # noqa: E402
    ClientUpdate,
    aggregate_client_updates,
    clip_delta_norm,
    delta_l2_norm,
    fedavg_aggregate,
)


def _update(base_sha: str, round_id: int, scale: float) -> ClientUpdate:
    return ClientUpdate(
        round_id=round_id, base_sha=base_sha,
        tensors={"w": torch.ones(4) * scale}, num_samples=10, adapter_mode="delta",
    )


def test_fedavg_refuses_lora_factor_averaging() -> None:
    a = {"layer.lora_A": torch.ones(2, 2), "layer.lora_B": torch.ones(2, 2)}
    b = {"layer.lora_A": torch.ones(2, 2) * 2, "layer.lora_B": torch.ones(2, 2) * 2}
    with pytest.raises(ValueError, match="LoRA"):
        fedavg_aggregate([a, b])


def test_aggregate_requires_shared_base_sha() -> None:
    with pytest.raises(ValueError, match="base_sha"):
        aggregate_client_updates([_update("shaX", 1, 1.0), _update("shaY", 1, 2.0)])


def test_aggregate_requires_shared_round_id() -> None:
    with pytest.raises(ValueError, match="round_id"):
        aggregate_client_updates([_update("shaX", 1, 1.0), _update("shaX", 2, 2.0)])


def test_aggregate_averages_matching_updates() -> None:
    agg = aggregate_client_updates([_update("shaX", 1, 2.0), _update("shaX", 1, 4.0)])
    assert torch.allclose(agg["w"], torch.ones(4) * 3.0)


def test_clip_delta_norm_bounds_influence() -> None:
    delta = {"w": torch.ones(4) * 10.0}  # L2 norm = 20
    assert delta_l2_norm(delta) == pytest.approx(20.0)
    clipped = clip_delta_norm(delta, max_norm=5.0)
    assert delta_l2_norm(clipped) == pytest.approx(5.0, abs=1e-4)
    # A one-off outlier client cannot dominate a small cohort after clipping.
    agg = aggregate_client_updates(
        [_update("s", 1, 1.0), _update("s", 1, 100.0)], max_delta_norm=5.0)
    assert delta_l2_norm({"w": agg["w"]}) <= 5.0 + 1e-4

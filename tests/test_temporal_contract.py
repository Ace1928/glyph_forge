"""Contract tests for exact, portable temporal rendering timelines."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from glyph_forge.temporal import (
    TEMPORAL_CONTRACT_VERSION,
    AudioPolicy,
    FrameRate,
    FrameRounding,
    TemporalContractError,
    TemporalRenderRequest,
)

_CONFORMANCE = json.loads(
    (Path(__file__).parent / "fixtures" / "temporal-contract-v1.json").read_text(
        encoding="utf-8"
    )
)


def test_temporal_request_round_trips_as_versioned_json() -> None:
    request = TemporalRenderRequest(
        start=2.5,
        duration=4.25,
        frame_rate="30000/1001",
        audio="discard",
        rounding="floor",
    )

    encoded = json.loads(json.dumps(request.to_dict()))
    restored = TemporalRenderRequest.from_dict(encoded)

    assert restored == request
    assert restored.contract_version == TEMPORAL_CONTRACT_VERSION
    assert restored.selected_frame_rate == FrameRate(30_000, 1_001)
    assert restored.audio_policy is AudioPolicy.DISCARD
    assert restored.rounding_policy is FrameRounding.FLOOR


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("30000/1001", FrameRate(30_000, 1_001)),
        ("29.97", FrameRate(2_997, 100)),
        (29.97002997002997, FrameRate(30_000, 1_001)),
        (30, FrameRate(30)),
        ({"numerator": 60, "denominator": 2}, FrameRate(30)),
    ],
)
def test_frame_rate_parses_and_normalizes_portable_values(
    value: object,
    expected: FrameRate,
) -> None:
    selected = FrameRate.parse(value)

    assert selected == expected
    assert FrameRate.from_dict(selected.to_dict()) == expected


@pytest.mark.parametrize(
    ("rounding", "expected"),
    [
        ("floor", 2),
        ("nearest", 3),
        ("ceil", 3),
    ],
)
def test_frame_alignment_has_explicit_deterministic_rounding(
    rounding: str,
    expected: int,
) -> None:
    timeline = TemporalRenderRequest(start=0.25, rounding=rounding).resolve(10)

    assert timeline.start_frame == expected


def test_resolved_timeline_uses_exact_frame_boundaries_for_audio_and_duration() -> None:
    timeline = TemporalRenderRequest(
        start=2.5,
        duration=4.25,
        frame_rate="30000/1001",
    ).resolve(24, source_frames=500)

    assert timeline.frame_rate == FrameRate(30_000, 1_001)
    assert timeline.start_frame == 75
    assert timeline.frame_count == 127
    assert timeline.end_frame == 202
    assert timeline.aligned_start == Fraction(1001, 400)
    assert timeline.encoded_duration == Fraction(127127, 30000)
    assert timeline.ffmpeg_start == "2.5025"
    assert timeline.ffmpeg_duration == "4.237566667"


def test_resolved_timeline_clamps_a_slice_to_available_source_frames() -> None:
    timeline = TemporalRenderRequest(start=9.8, duration=4).resolve(
        10,
        source_frames=100,
    )

    assert timeline.start_frame == 98
    assert timeline.frame_count == 2
    assert timeline.end_frame == 100
    assert timeline.encoded_seconds == 0.2


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"contract_version": 99}, "contract version"),
        ({"start": True}, "start must be a number"),
        ({"start": -1}, "non-negative"),
        ({"duration": 0}, "positive"),
        ({"frame_rate": "0/1"}, "numerator"),
        ({"frame_rate": "not-a-rate"}, "number or ratio"),
        ({"audio": "copy"}, "audio policy"),
        ({"rounding": "sometimes"}, "frame rounding"),
    ],
)
def test_temporal_request_rejects_ambiguous_or_unsafe_values(
    updates: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(TemporalContractError, match=message):
        TemporalRenderRequest(**updates)


def test_temporal_request_wraps_unknown_serialized_fields() -> None:
    with pytest.raises(TemporalContractError, match="Malformed serialized"):
        TemporalRenderRequest.from_dict({"surprise": True})


@pytest.mark.parametrize(
    "case",
    _CONFORMANCE["cases"],
    ids=[case["name"] for case in _CONFORMANCE["cases"]],
)
def test_native_timeline_matches_checked_in_contract_fixture(
    case: dict[str, Any],
) -> None:
    request = TemporalRenderRequest.from_dict(case["request"])
    timeline = request.resolve(case["source_rate"], case["source_frames"])

    values = timeline.to_dict()
    for key, expected in case["expected"].items():
        assert values[key] == pytest.approx(expected)

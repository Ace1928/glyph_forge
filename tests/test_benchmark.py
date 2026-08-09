"""Benchmark and built-in showcase tests."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from glyph_forge.benchmark import benchmark_renderers, synthetic_frame
from glyph_forge.cli import app


def test_synthetic_frame_is_deterministic() -> None:
    first = synthetic_frame(24, 12)
    second = synthetic_frame(24, 12)

    assert first.shape == (12, 24, 3)
    assert first.dtype.name == "uint8"
    assert (first == second).all()


def test_single_renderer_benchmark_returns_metrics() -> None:
    result = benchmark_renderers(
        "eco",
        modes=["edge"],
        iterations=1,
        warmup=0,
    )[0]

    assert result.mode == "edge"
    assert result.milliseconds > 0
    assert result.frames_per_second > 0
    assert result.output_bytes > 0


def test_benchmark_cli_can_emit_machine_readable_json() -> None:
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--mode",
            "glyph",
            "--performance",
            "eco",
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["mode"] == "glyph"


def test_demo_needs_no_external_file() -> None:
    result = CliRunner().invoke(
        app,
        ["demo", "--mode", "braille", "--width", "20", "--no-color"],
    )

    assert result.exit_code == 0, result.output
    assert "Glyph Forge · braille" in result.output


@pytest.mark.parametrize("value", [(0, 1), (1, -1)])
def test_benchmark_rejects_invalid_counts(value: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        benchmark_renderers(iterations=value[0], warmup=value[1])

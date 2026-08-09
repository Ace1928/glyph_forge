"""Small reproducible renderer benchmarks used by the CLI and maintainers."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from .live.renderers import (
    FrameRenderer,
    PluginRenderMode,
    RenderConfig,
    RenderMode,
    normalize_render_mode,
)
from .runtime import RuntimeProfile, detect_runtime_profile


@dataclass(frozen=True, slots=True)
class RendererBenchmark:
    """Measured latency and throughput for one render mode."""

    mode: str
    source_width: int
    source_height: int
    columns: int
    rows: int
    iterations: int
    milliseconds: float
    frames_per_second: float
    output_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def synthetic_frame(width: int, height: int) -> np.ndarray:
    """Create a deterministic RGB frame with gradients and hard edges."""

    if width < 1 or height < 1:
        raise ValueError("Synthetic frame dimensions must be positive")
    y, x = np.indices((height, width), dtype=np.float32)
    x_norm = x / max(1, width - 1)
    y_norm = y / max(1, height - 1)
    rings = (np.sin(np.hypot(x_norm - 0.5, y_norm - 0.5) * 52) + 1) * 0.5
    return np.stack(
        (
            x_norm * 255,
            y_norm * 255,
            rings * 255,
        ),
        axis=2,
    ).astype(np.uint8)


def _source_dimensions(profile: RuntimeProfile) -> tuple[int, int]:
    if profile.tier.value == "eco":
        return 640, 360
    if profile.tier.value == "workstation":
        return 1920, 1080
    return 1280, 720


def benchmark_renderers(
    preference: str = "auto",
    *,
    modes: Iterable[RenderMode | PluginRenderMode | str] | None = None,
    iterations: int = 3,
    warmup: int = 1,
) -> list[RendererBenchmark]:
    """Measure selected hot paths using the active adaptive profile."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")
    profile = detect_runtime_profile(preference)
    source_width, source_height = _source_dimensions(profile)
    frame = synthetic_frame(source_width, source_height)
    selected_modes = list(modes or RenderMode)
    results: list[RendererBenchmark] = []

    for value in selected_modes:
        mode = normalize_render_mode(value)
        color = "ansi256" if mode is RenderMode.HALF_BLOCK else "none"
        renderer = FrameRenderer(
            RenderConfig(
                width=profile.stream_width,
                mode=mode,
                color=color,
                charset="detailed",
                edge_algorithm="scharr",
                resample=profile.resample,
            )
        )
        result = None
        for _ in range(warmup):
            result = renderer.render(frame)
        started = time.perf_counter()
        for _ in range(iterations):
            result = renderer.render(frame)
        elapsed = time.perf_counter() - started
        assert result is not None
        average = elapsed / iterations
        results.append(
            RendererBenchmark(
                mode=mode.value,
                source_width=source_width,
                source_height=source_height,
                columns=result.width,
                rows=result.height,
                iterations=iterations,
                milliseconds=average * 1000,
                frames_per_second=1 / average if average else float("inf"),
                output_bytes=len(result.text.encode("utf-8")),
            )
        )
    return results


__all__ = ["RendererBenchmark", "benchmark_renderers", "synthetic_frame"]

#!/usr/bin/env python3
"""Benchmark the Glyph Forge streaming renderer with generated frames."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from typing import Any

import numpy as np

from glyph_forge import __version__

SOURCE_SIZES = {
    "360p": (640, 360),
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure in-memory Glyph Forge frame rendering.",
        epilog="Example: python scripts/benchmark.py --iterations 100 --mode ansi256",
    )
    parser.add_argument("--source", choices=SOURCE_SIZES, default="720p")
    parser.add_argument("--output-width", type=int, default=120)
    parser.add_argument("--output-height", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--mode", choices=["none", "ansi256", "truecolor"], default="ansi256")
    parser.add_argument(
        "--glyph-mode",
        choices=["ascii", "blocks", "braille", "gradient", "hybrid"],
        default="gradient",
    )
    parser.add_argument("--seed", type=int, default=1928)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def positive(value: int, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if value < minimum:
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparison}")
    return value


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            'OpenCV is required; install with: pip install "glyph-forge[streaming]"'
        ) from exc

    from glyph_forge.streaming.core.renderer import GlyphRenderer, RenderConfig

    iterations = positive(args.iterations, "iterations")
    warmup = positive(args.warmup, "warmup", allow_zero=True)
    output_width = positive(args.output_width, "output-width")
    output_height = positive(args.output_height, "output-height")
    source_width, source_height = SOURCE_SIZES[args.source]

    generator = np.random.default_rng(args.seed)
    frame = generator.integers(
        0,
        256,
        size=(source_height, source_width, 3),
        dtype=np.uint8,
    )
    renderer = GlyphRenderer(RenderConfig(mode=args.glyph_mode, color=args.mode))

    for _ in range(warmup):
        renderer.render(frame, width=output_width, height=output_height)

    samples_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        renderer.render(frame, width=output_width, height=output_height)
        samples_ms.append((time.perf_counter() - started) * 1000.0)

    mean_ms = statistics.fmean(samples_ms)
    return {
        "glyph_forge": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "source": [source_width, source_height],
        "output": [output_width, output_height],
        "glyph_mode": args.glyph_mode,
        "color_mode": args.mode,
        "iterations": iterations,
        "warmup": warmup,
        "seed": args.seed,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(samples_ms),
        "p95_ms": percentile(samples_ms, 0.95),
        "standard_deviation_ms": statistics.pstdev(samples_ms),
        "estimated_fps": 1000.0 / mean_ms,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"Glyph Forge {report['glyph_forge']} renderer benchmark")
    print(f"Platform: {report['platform']} | Python {report['python']}")
    print(f"NumPy {report['numpy']} | OpenCV {report['opencv']}")
    print(
        f"Source: {report['source'][0]}x{report['source'][1]} | "
        f"Output: {report['output'][0]}x{report['output'][1]}"
    )
    print(f"Mode: {report['glyph_mode']} | Color: {report['color_mode']}")
    print(f"Iterations: {report['iterations']} after {report['warmup']} warm-up frames")
    print(f"Median: {report['median_ms']:.3f} ms | Mean: {report['mean_ms']:.3f} ms")
    print(f"P95: {report['p95_ms']:.3f} ms | Std dev: {report['standard_deviation_ms']:.3f} ms")
    print(f"Estimated throughput: {report['estimated_fps']:.2f} frames/second")


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run_benchmark(args)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

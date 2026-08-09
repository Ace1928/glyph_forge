"""Runnable in-process example for the Glyph Forge plugin API v1."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from glyph_forge import FrameRenderer, RenderConfig, create_frame_source
from glyph_forge.plugins import (
    ExportReceipt,
    ExportRequest,
    PluginManifest,
    RendererRequest,
    RenderOutput,
    SourceRequest,
    TransformRequest,
    register_plugin,
)


class GradientSource:
    """Finite generated media source requiring no camera or video backend."""

    name = "example:animated-gradient"

    def __init__(self, request: SourceRequest) -> None:
        self.width = request.width or 320
        self.height = request.height or 180
        self.remaining = 3
        self.phase = 0

    def read(self) -> np.ndarray | None:
        if self.remaining == 0:
            return None
        self.remaining -= 1
        y, x = np.indices((self.height, self.width), dtype=np.uint16)
        frame = np.stack(
            (
                (x + self.phase) % 256,
                (y * 2 + self.phase) % 256,
                (x + y + self.phase * 3) % 256,
            ),
            axis=2,
        ).astype(np.uint8)
        self.phase += 24
        return frame

    def close(self) -> None:
        self.remaining = 0


class OutlineRenderer:
    """Compose an extension from the maintained vectorized edge renderer."""

    def __init__(self, request: RendererRequest) -> None:
        self.inner = FrameRenderer(
            replace(request.config, mode="edge", edge_algorithm="scharr")
        )

    def render(
        self,
        frame: Any,
        *,
        max_width: int | None = None,
        max_height: int | None = None,
    ) -> RenderOutput:
        result = self.inner.render(
            frame,
            max_width=max_width,
            max_height=max_height,
        )
        return RenderOutput(result.text, result.width, result.height)


def invert(request: TransformRequest) -> np.ndarray:
    """Example explicitly invoked transform."""

    return 255 - np.asarray(request.source, dtype=np.uint8)


def export_text(request: ExportRequest) -> ExportReceipt:
    """Example exporter with a structured receipt."""

    output = Path(request.destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(str(request.source), encoding="utf-8")
    return ExportReceipt(output, media_type="text/plain")


def plugin() -> PluginManifest:
    """Entry point exported by a distributable plugin package."""

    return PluginManifest(
        name="Glyph Forge example",
        version="1.0.0",
        description="Generated frames and a composed outline renderer",
        sources={"gradient": GradientSource},
        renderers={"outline": OutlineRenderer},
        transforms={"invert": invert},
        exporters={"text": export_text},
    )


def main() -> None:
    """Register in-process, then exercise normal source and render dispatch."""

    register_plugin("example", plugin())
    source = create_frame_source("plugin:example/gradient")
    frame = source.read()
    assert frame is not None
    result = FrameRenderer(
        RenderConfig(width=48, mode="plugin:example/outline")
    ).render(frame)
    print(result.text)
    source.close()


if __name__ == "__main__":
    main()

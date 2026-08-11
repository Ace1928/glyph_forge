"""High-throughput full-colour glyph video rendering.

Frames are sampled and rendered in memory, then streamed directly to FFmpeg.
No image sequence is written to disk and the source audio is muxed into the
finished file when present.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Generator, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from ..runtime import (
    PerformanceTier,
    detect_runtime_profile,
    python_install_hint,
    subprocess_environment,
)
from ..utils.alphabet_manager import AlphabetCategory, AlphabetManager
from ..visual import (
    DEFAULT_BRIGHTNESS,
    DEFAULT_CONTRAST,
    apply_tone,
    normalize_tone,
)

RGBFrame = NDArray[np.uint8]
ProgressCallback = Callable[["VideoExportProgress"], None]


class VideoExportError(RuntimeError):
    """Raised when a glyph video cannot be rendered or encoded."""


class MissingMediaDependency(VideoExportError):
    """Raised when an optional video dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class VideoExportConfig:
    """Options for a streamed glyph-video export."""

    width: int = 1920
    height: int = 1080
    columns: int = 160
    rows: int = 90
    charset: str = "detailed"
    font: str | os.PathLike[str] | None = None
    start: float = 0.0
    duration: float | None = None
    crf: int = 18
    preset: str = "veryfast"
    ffmpeg: str = "ffmpeg"
    workers: int = 1
    brightness: float = DEFAULT_BRIGHTNESS
    contrast: float = DEFAULT_CONTRAST

    @classmethod
    def adaptive(
        cls,
        preference: str = "auto",
        **overrides: Any,
    ) -> "VideoExportConfig":
        """Choose divisible dimensions suited to the detected hardware."""

        profile = detect_runtime_profile(preference)
        defaults = {
            PerformanceTier.ECO: (640, 360, 80, 45),
            PerformanceTier.BALANCED: (1280, 720, 128, 72),
            PerformanceTier.WORKSTATION: (1920, 1080, 160, 90),
        }
        width, height, columns, rows = defaults[profile.tier]
        values: dict[str, Any] = {
            "width": width,
            "height": height,
            "columns": columns,
            "rows": rows,
            "workers": profile.workers,
        }
        values.update(overrides)
        return cls(**values)

    @property
    def cell_width(self) -> int:
        return math.ceil(self.width / self.columns)

    @property
    def cell_height(self) -> int:
        return math.ceil(self.height / self.rows)

    @property
    def uses_integral_cells(self) -> bool:
        """Whether glyph cells map to output pixels without a final resize."""

        return self.width % self.columns == 0 and self.height % self.rows == 0

    def validated(self) -> "VideoExportConfig":
        """Validate all constraints and return this immutable configuration."""

        if self.width < 2 or self.height < 2:
            raise ValueError("Video width and height must be at least two pixels")
        if self.width % 2 or self.height % 2:
            raise ValueError("Video width and height must be even for yuv420p output")
        if self.columns < 1 or self.rows < 1:
            raise ValueError("Glyph columns and rows must be positive")
        if self.start < 0:
            raise ValueError("Start time cannot be negative")
        if self.duration is not None and self.duration <= 0:
            raise ValueError("Duration must be greater than zero")
        if not 0 <= self.crf <= 51:
            raise ValueError("CRF must be between 0 and 51")
        if not self.preset.strip():
            raise ValueError("FFmpeg preset cannot be empty")
        if not self.ffmpeg.strip():
            raise ValueError("FFmpeg executable cannot be empty")
        if not 1 <= self.workers <= 64:
            raise ValueError("Worker count must be between 1 and 64")
        normalized_brightness = normalize_tone(self.brightness, name="brightness")
        normalized_contrast = normalize_tone(self.contrast, name="contrast")
        if not _resolve_charset(self.charset):
            raise ValueError("Charset cannot be empty")
        if (
            normalized_brightness != self.brightness
            or normalized_contrast != self.contrast
        ):
            return replace(
                self,
                brightness=normalized_brightness,
                contrast=normalized_contrast,
            )
        return self


@dataclass(frozen=True, slots=True)
class VideoExportProgress:
    """Progress snapshot emitted while a video is being rendered."""

    rendered_frames: int
    total_frames: int | None
    fps: float
    elapsed: float

    @property
    def fraction(self) -> float | None:
        if not self.total_frames:
            return None
        return min(1.0, self.rendered_frames / self.total_frames)


@dataclass(frozen=True, slots=True)
class VideoExportResult:
    """Summary of a completed glyph-video export."""

    output: Path
    rendered_frames: int
    fps: float
    elapsed: float
    width: int
    height: int
    columns: int
    rows: int
    workers: int = 1
    source: Path | None = None
    source_bytes: int = 0
    output_bytes: int = 0

    @property
    def rendered_seconds(self) -> float:
        """Duration represented by the encoded frames."""

        return self.rendered_frames / self.fps if self.fps > 0 else 0.0

    @property
    def render_fps(self) -> float:
        """Frames rendered per wall-clock second."""

        return self.rendered_frames / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def realtime_factor(self) -> float:
        """Wall time divided by encoded duration (below one is real-time)."""

        duration = self.rendered_seconds
        return self.elapsed / duration if duration > 0 else 0.0

    @property
    def glyph_cells_per_second(self) -> float:
        """Glyph samples processed per wall-clock second."""

        return self.render_fps * self.columns * self.rows

    @property
    def output_megapixels_per_second(self) -> float:
        """Rendered raster megapixels produced per wall-clock second."""

        return self.render_fps * self.width * self.height / 1_000_000

    @property
    def raw_rgb_bytes(self) -> int:
        """Bytes streamed to FFmpeg before compression."""

        return self.rendered_frames * self.width * self.height * 3

    @property
    def output_source_ratio(self) -> float | None:
        """Encoded output size relative to the source file."""

        return self.output_bytes / self.source_bytes if self.source_bytes else None

    def to_dict(self) -> dict[str, Any]:
        """Return complete JSON-ready performance and output metrics."""

        return {
            "source": str(self.source) if self.source is not None else None,
            "output": str(self.output),
            "rendered_frames": self.rendered_frames,
            "source_fps": self.fps,
            "render_fps": self.render_fps,
            "rendered_seconds": self.rendered_seconds,
            "elapsed_seconds": self.elapsed,
            "realtime_factor": self.realtime_factor,
            "width": self.width,
            "height": self.height,
            "columns": self.columns,
            "rows": self.rows,
            "workers": self.workers,
            "glyph_cells_per_second": self.glyph_cells_per_second,
            "output_megapixels_per_second": self.output_megapixels_per_second,
            "source_bytes": self.source_bytes,
            "output_bytes": self.output_bytes,
            "output_source_ratio": self.output_source_ratio,
            "raw_rgb_bytes": self.raw_rgb_bytes,
        }


def _resolve_charset(name_or_characters: str) -> str:
    """Resolve a named alphabet while still allowing literal custom glyphs."""

    known = set(AlphabetManager.list_available_alphabets())
    known.update(AlphabetManager.list_special_sets())
    known.update(AlphabetManager.list_by_category(AlphabetCategory.LANGUAGES))
    if name_or_characters in known:
        return AlphabetManager.get_alphabet(name_or_characters)
    return name_or_characters


def _font_candidates(explicit: str | os.PathLike[str] | None) -> tuple[str, ...]:
    candidates: list[str] = []
    if explicit is not None:
        candidates.append(os.fspath(explicit))
    else:
        configured = os.environ.get("GLYPH_FORGE_FONT")
        if configured:
            candidates.append(configured)
        prefix = os.environ.get("PREFIX")
        if prefix:
            candidates.extend(
                [
                    str(Path(prefix) / "share/fonts/TTF/DejaVuSansMono-Bold.ttf"),
                    str(
                        Path(prefix)
                        / "share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
                    ),
                ]
            )
        candidates.extend(
            [
                "DejaVuSansMono-Bold.ttf",
                "DejaVuSansMono.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf",
                "/System/Library/Fonts/Menlo.ttc",
                "/System/Library/Fonts/Monaco.ttf",
                "C:/Windows/Fonts/consolab.ttf",
                "C:/Windows/Fonts/consola.ttf",
                "C:/Windows/Fonts/lucon.ttf",
            ]
        )
    return tuple(dict.fromkeys(candidates))


def find_monospace_font(
    explicit: str | os.PathLike[str] | None = None,
    *,
    size: int = 12,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, str]:
    """Load a portable monospace font and return it with its identifier."""

    errors: list[str] = []
    for candidate in _font_candidates(explicit):
        try:
            return ImageFont.truetype(candidate, size), candidate
        except (OSError, TypeError) as exc:
            errors.append(f"{candidate}: {exc}")
    if explicit is not None:
        detail = errors[-1] if errors else os.fspath(explicit)
        raise VideoExportError(f"Could not load font ({detail})")

    # Pillow's bundled fallback is always available. It is less expressive but
    # keeps video export portable in minimal containers and fresh installations.
    try:
        return ImageFont.load_default(size=size), "Pillow default"
    except TypeError:  # Pillow < 10.1 does not accept the size keyword.
        return ImageFont.load_default(), "Pillow default"


@lru_cache(maxsize=12)
def _cached_glyph_atlas(
    charset: str,
    font_identifier: str | None,
    cell_width: int,
    cell_height: int,
) -> RGBFrame:
    font_size = max(8, int(cell_height * 1.12))
    font, _ = find_monospace_font(font_identifier, size=font_size)
    atlas = np.empty((len(charset), cell_height, cell_width), dtype=np.uint8)
    for index, character in enumerate(charset):
        image = Image.new("L", (cell_width, cell_height), 0)
        draw = ImageDraw.Draw(image)
        box = draw.textbbox((0, 0), character, font=font, stroke_width=0)
        glyph_width = box[2] - box[0]
        glyph_height = box[3] - box[1]
        x = (cell_width - glyph_width) // 2 - box[0]
        y = (cell_height - glyph_height) // 2 - box[1]
        draw.text((x, y), character, fill=255, font=font)
        atlas[index] = np.asarray(image, dtype=np.uint8)
    atlas.flags.writeable = False
    return atlas


def glyph_atlas(
    charset: str,
    font: str | os.PathLike[str] | None,
    cell_width: int,
    cell_height: int,
) -> RGBFrame:
    """Build (and cache) alpha masks for every glyph in a character set."""

    if not charset:
        raise ValueError("Charset cannot be empty")
    if cell_width < 1 or cell_height < 1:
        raise ValueError("Glyph cell dimensions must be positive")
    identifier = os.fspath(font) if font is not None else None
    return _cached_glyph_atlas(charset, identifier, cell_width, cell_height)


class GlyphVideoRenderer:
    """Reusable vectorized renderer for full-colour video frames."""

    def __init__(self, config: VideoExportConfig | None = None) -> None:
        self.config = (config or VideoExportConfig()).validated()
        self.charset = _resolve_charset(self.config.charset)
        self.atlas = glyph_atlas(
            self.charset,
            self.config.font,
            self.config.cell_width,
            self.config.cell_height,
        )

    def render_sampled_rgb(self, sampled: NDArray[Any]) -> RGBFrame:
        """Render an RGB array containing one source colour per glyph cell."""

        rgb = np.asarray(sampled)
        expected = (self.config.rows, self.config.columns, 3)
        if rgb.shape != expected:
            raise ValueError(
                f"Sampled frame must have shape {expected}, got {rgb.shape}"
            )
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = apply_tone(rgb, self.config.brightness, self.config.contrast)

        pixels = rgb.astype(np.uint16, copy=False)
        gray = (
            pixels[:, :, 0] * 77 + pixels[:, :, 1] * 150 + pixels[:, :, 2] * 29
        ) >> 8
        indices = np.minimum(
            (gray * len(self.atlas)) // 256,
            len(self.atlas) - 1,
        )
        masks = self.atlas[indices]
        coloured = (
            masks[..., None].astype(np.uint16) * pixels[:, :, None, None, :]
        ) // 255
        rendered = cast(
            RGBFrame,
            coloured.transpose(0, 2, 1, 3, 4)
            .reshape(
                self.config.rows * self.config.cell_height,
                self.config.columns * self.config.cell_width,
                3,
            )
            .astype(np.uint8),
        )
        if self.config.uses_integral_cells:
            return rendered
        # Arbitrary output pixels remain independent from the sampling grid.
        # Divisible configurations keep the zero-copy hot path; uncommon
        # fractional cell sizes receive one high-quality final resample.
        return np.asarray(
            Image.fromarray(rendered, mode="RGB").resize(
                (self.config.width, self.config.height),
                Image.Resampling.LANCZOS,
            ),
            dtype=np.uint8,
        )

    def render_bgr(self, frame: NDArray[Any], cv2: Any | None = None) -> RGBFrame:
        """Sample and render an OpenCV-style BGR source frame."""

        backend = cv2 or _load_opencv()
        sampled = backend.resize(
            frame,
            (self.config.columns, self.config.rows),
            interpolation=backend.INTER_AREA,
        )
        return self.render_sampled_rgb(np.asarray(sampled)[:, :, ::-1])


def build_ffmpeg_command(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    config: VideoExportConfig,
    fps: float,
) -> list[str]:
    """Build the shell-free FFmpeg command used by the streaming exporter."""

    selected = config.validated()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be a positive finite number")
    command = [
        selected.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{selected.width}x{selected.height}",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        "pipe:0",
        "-ss",
        f"{selected.start:.6f}",
        "-i",
        os.fspath(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        selected.preset,
        "-crf",
        str(selected.crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
    ]
    if selected.duration is not None:
        command.extend(["-t", f"{selected.duration:.6f}"])
    command.append(os.fspath(output_path))
    return command


def _load_opencv() -> Any:
    try:
        import cv2  # type: ignore[import-untyped]
    except (ImportError, OSError) as exc:
        raise MissingMediaDependency(
            f"Video export requires OpenCV; {python_install_hint('media')}"
        ) from exc
    return cv2


def _executable_available(executable: str) -> bool:
    if Path(executable).parent != Path("."):
        return Path(executable).is_file()
    return shutil.which(executable) is not None


def _partial_output_path(output: Path) -> Path:
    suffix = output.suffix or ".mp4"
    nonce = f"{os.getpid()}-{time.monotonic_ns()}"
    return output.with_name(f".{output.stem}.glyph-forge-{nonce}.partial{suffix}")


def _iter_rendered_frames(
    capture: Any,
    renderer: GlyphVideoRenderer,
    backend: Any,
    *,
    total_frames: int | None,
    workers: int,
) -> Generator[RGBFrame, None, None]:
    """Render captured frames in order with at most one frame per worker queued."""

    if workers == 1:
        submitted = 0
        while total_frames is None or submitted < total_frames:
            ok, frame = capture.read()
            if not ok:
                break
            submitted += 1
            yield renderer.render_bgr(frame, backend)
        return

    submitted = 0
    source_ended = False
    pending: deque[Future[RGBFrame]] = deque()
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="glyph-forge-video",
    ) as executor:
        while pending or not source_ended:
            while not source_ended and len(pending) < workers:
                if total_frames is not None and submitted >= total_frames:
                    source_ended = True
                    break
                ok, frame = capture.read()
                if not ok:
                    source_ended = True
                    break
                pending.append(executor.submit(renderer.render_bgr, frame, backend))
                submitted += 1
            if pending:
                # Futures stay in capture order even when later frames finish
                # first, so audio timing and frame order remain exact.
                yield pending.popleft().result()


def export_glyph_video(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    config: VideoExportConfig | None = None,
    *,
    progress: ProgressCallback | None = None,
) -> VideoExportResult:
    """Stream a full-colour glyph rendering to FFmpeg with source audio."""

    selected = (config or VideoExportConfig()).validated()
    source = Path(input_path).expanduser()
    output = Path(output_path).expanduser()
    if not source.is_file():
        raise VideoExportError(f"Input does not exist: {source}")
    source_bytes = source.stat().st_size
    if source.resolve() == output.resolve():
        raise VideoExportError("Input and output paths must be different")
    if not _executable_available(selected.ffmpeg):
        raise MissingMediaDependency(
            f"FFmpeg executable was not found: {selected.ffmpeg!r}"
        )

    cv2 = _load_opencv()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise VideoExportError(f"OpenCV could not open: {source}")

    started = time.monotonic()
    encoder: subprocess.Popen[bytes] | None = None
    partial = _partial_output_path(output)
    rendered = 0
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = (
            source_fps if math.isfinite(source_fps) and 1 <= source_fps <= 120 else 30.0
        )
        total_source_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        start_frame = max(0, int(round(selected.start * fps)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if selected.duration is not None:
            total_frames: int | None = max(1, int(round(selected.duration * fps)))
        elif total_source_frames > 0:
            total_frames = max(0, total_source_frames - start_frame)
        else:
            total_frames = None

        output.parent.mkdir(parents=True, exist_ok=True)
        partial.unlink(missing_ok=True)
        renderer = GlyphVideoRenderer(selected)
        command = build_ffmpeg_command(source, partial, selected, fps)
        try:
            encoder = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                env=subprocess_environment(),
            )
        except OSError as exc:
            raise VideoExportError(f"Could not start FFmpeg: {exc}") from exc
        if encoder.stdin is None:  # pragma: no cover - defensive subprocess guard
            raise VideoExportError("FFmpeg did not provide an input pipe")

        pipe_broken = False
        rendered_frames = _iter_rendered_frames(
            capture,
            renderer,
            cv2,
            total_frames=total_frames,
            workers=selected.workers,
        )
        try:
            for output_frame in rendered_frames:
                encoder.stdin.write(output_frame.tobytes())
                rendered += 1
                if progress is not None:
                    progress(
                        VideoExportProgress(
                            rendered_frames=rendered,
                            total_frames=total_frames,
                            fps=fps,
                            elapsed=time.monotonic() - started,
                        )
                    )
        except BrokenPipeError:
            pipe_broken = True
        finally:
            rendered_frames.close()

        if rendered == 0:
            raise VideoExportError("The input did not yield any video frames")
        try:
            encoder.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        status = encoder.wait()
        if status:
            raise VideoExportError(f"FFmpeg exited with status {status}")
        if pipe_broken:
            raise VideoExportError("FFmpeg closed its input before export completed")
        os.replace(partial, output)
    except BaseException:
        if encoder is not None and encoder.poll() is None:
            if encoder.stdin is not None and not encoder.stdin.closed:
                try:
                    encoder.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            try:
                encoder.terminate()
                encoder.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                encoder.kill()
                encoder.wait()
        partial.unlink(missing_ok=True)
        raise
    finally:
        capture.release()

    return VideoExportResult(
        output=output,
        rendered_frames=rendered,
        fps=fps,
        elapsed=time.monotonic() - started,
        width=selected.width,
        height=selected.height,
        columns=selected.columns,
        rows=selected.rows,
        workers=selected.workers,
        source=source,
        source_bytes=source_bytes,
        output_bytes=output.stat().st_size,
    )


def with_video_overrides(
    config: VideoExportConfig,
    **overrides: Any,
) -> VideoExportConfig:
    """Return a config with only non-``None`` override values applied."""

    return replace(
        config, **{key: value for key, value in overrides.items() if value is not None}
    )


__all__ = [
    "GlyphVideoRenderer",
    "MissingMediaDependency",
    "ProgressCallback",
    "VideoExportConfig",
    "VideoExportError",
    "VideoExportProgress",
    "VideoExportResult",
    "build_ffmpeg_command",
    "export_glyph_video",
    "find_monospace_font",
    "glyph_atlas",
    "with_video_overrides",
]

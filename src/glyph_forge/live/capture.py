"""Portable, bounded-latency frame capture backends.

Capture backends expose one small protocol. :class:`LatestFramePump` runs a
source on a producer thread and stores only its newest frame, deliberately
dropping stale work so interactive displays remain responsive under load.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

RGBFrame = NDArray[np.uint8]


class CaptureError(RuntimeError):
    """Raised when a live source cannot be opened or read."""


class CaptureBackendUnavailable(CaptureError):
    """Raised when an optional capture backend cannot be initialized."""


@runtime_checkable
class FrameSource(Protocol):
    """Minimal interface implemented by every live capture backend."""

    @property
    def name(self) -> str:
        """Human-readable source description."""

    def read(self) -> RGBFrame | None:
        """Return the next RGB frame, or ``None`` when the source ends."""

    def close(self) -> None:
        """Release native resources. Calling this more than once is safe."""


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    """The newest frame plus monotonic capture metadata."""

    pixels: RGBFrame
    sequence: int
    captured_at: float


def _as_rgb_frame(frame: NDArray[Any]) -> RGBFrame:
    pixels = np.asarray(frame)
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[:, :, None], 3, axis=2)
    if pixels.ndim != 3 or pixels.shape[2] not in {3, 4}:
        raise CaptureError(
            "Capture frames must have shape (height, width), (..., 3), or (..., 4)"
        )
    pixels = pixels[:, :, :3]
    if pixels.dtype != np.uint8:
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(pixels)


def _load_opencv() -> Any:
    try:
        import cv2  # type: ignore[import-untyped]
    except (ImportError, OSError) as exc:
        raise CaptureError(
            "Camera and video capture require OpenCV; install glyph-forge[media]"
        ) from exc
    return cv2


class _FramePacer:
    def __init__(self, fps: float | None) -> None:
        self.interval = 1 / fps if fps is not None and fps > 0 else 0.0
        self.next_due: float | None = None

    def wait(self) -> None:
        if not self.interval:
            return
        now = time.monotonic()
        if self.next_due is None:
            self.next_due = now
        if self.next_due > now:
            time.sleep(self.next_due - now)
            now = time.monotonic()
        self.next_due = max(self.next_due + self.interval, now)


class OpenCVFrameSource:
    """Webcam or video-file capture backed by OpenCV."""

    def __init__(
        self,
        source: int | str | Path = 0,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        loop: bool = False,
    ) -> None:
        self._cv2 = _load_opencv()
        self._source = source
        self._camera = isinstance(source, int)
        self._loop = loop and not self._camera
        self._closed = False
        self._lock = threading.Lock()
        native_source: int | str = source if isinstance(source, int) else str(source)
        self._capture = self._cv2.VideoCapture(native_source)
        if not self._capture.isOpened():
            self._capture.release()
            kind = "camera" if self._camera else "video"
            raise CaptureError(f"OpenCV could not open {kind}: {source}")

        if self._camera:
            if width is not None:
                self._capture.set(self._cv2.CAP_PROP_FRAME_WIDTH, width)
            if height is not None:
                self._capture.set(self._cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps is not None:
                self._capture.set(self._cv2.CAP_PROP_FPS, fps)
            pace = None  # Camera reads are paced by the device/backend.
        else:
            source_fps = float(self._capture.get(self._cv2.CAP_PROP_FPS))
            pace = fps or (source_fps if 1 <= source_fps <= 120 else 30.0)
        self._pacer = _FramePacer(pace)

    @property
    def name(self) -> str:
        if self._camera:
            return f"camera:{self._source}"
        return str(self._source)

    def read(self) -> RGBFrame | None:
        self._pacer.wait()
        with self._lock:
            if self._closed:
                return None
            ok, frame = self._capture.read()
            if not ok and self._loop:
                self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._capture.read()
            if not ok:
                return None
        # BGR -> RGB. A contiguous copy avoids negative-stride surprises later.
        return np.ascontiguousarray(np.asarray(frame)[:, :, ::-1])

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._capture.release()


class MSSScreenSource:
    """Cross-platform screen capture backed by the optional MSS package."""

    def __init__(
        self,
        monitor: int = 1,
        *,
        fps: float = 30,
        region: tuple[int, int, int, int] | None = None,
    ) -> None:
        try:
            import mss  # type: ignore[import-untyped]
        except (ImportError, OSError) as exc:
            raise CaptureBackendUnavailable(
                "Screen capture needs MSS; install glyph-forge[media]"
            ) from exc
        try:
            self._mss = mss.mss()
        except Exception as exc:
            raise CaptureBackendUnavailable(
                f"MSS could not initialize screen capture: {exc}"
            ) from exc
        monitors = self._mss.monitors
        if monitor < 0 or monitor >= len(monitors):
            self._mss.close()
            raise CaptureError(
                f"Monitor {monitor} is unavailable; choose 0-{len(monitors) - 1}"
            )
        if region is None:
            self._bounds: dict[str, int] = dict(monitors[monitor])
        else:
            left, top, width, height = region
            if width < 1 or height < 1:
                self._mss.close()
                raise CaptureError(
                    "Screen capture region must have positive dimensions"
                )
            self._bounds = {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            }
        self._monitor = monitor
        self._closed = False
        self._pacer = _FramePacer(fps)

    @property
    def name(self) -> str:
        return f"screen:{self._monitor}"

    def read(self) -> RGBFrame | None:
        if self._closed:
            return None
        self._pacer.wait()
        try:
            bgra = np.asarray(self._mss.grab(self._bounds), dtype=np.uint8)
        except Exception as exc:
            raise CaptureError(f"Screen capture failed: {exc}") from exc
        return np.ascontiguousarray(bgra[:, :, 2::-1])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._mss.close()


class PillowScreenSource:
    """Dependency-free screen-capture fallback for supported desktop systems."""

    def __init__(
        self,
        *,
        fps: float = 15,
        region: tuple[int, int, int, int] | None = None,
    ) -> None:
        try:
            from PIL import ImageGrab
        except ImportError as exc:  # pragma: no cover - part of normal Pillow wheels
            raise CaptureError(
                "This Pillow build has no screen-capture backend"
            ) from exc
        self._grab = ImageGrab.grab
        self._region = region
        self._closed = False
        self._pacer = _FramePacer(fps)

    @property
    def name(self) -> str:
        return "screen:pillow"

    def read(self) -> RGBFrame | None:
        if self._closed:
            return None
        self._pacer.wait()
        bbox = None
        if self._region is not None:
            left, top, width, height = self._region
            bbox = (left, top, left + width, top + height)
        try:
            image = self._grab(bbox=bbox, all_screens=True).convert("RGB")
        except Exception as exc:
            raise CaptureError(
                "Pillow screen capture failed; install glyph-forge[media] for MSS "
                f"or grant screen-recording permission ({exc})"
            ) from exc
        return np.asarray(image, dtype=np.uint8)

    def close(self) -> None:
        self._closed = True


class IterableFrameSource:
    """Small adapter useful for extensions, tests, and generated frame streams."""

    def __init__(self, frames: Iterable[NDArray[Any]], name: str = "iterable") -> None:
        self._frames = iter(frames)
        self._name = name
        self._closed = False

    @property
    def name(self) -> str:
        return self._name

    def read(self) -> RGBFrame | None:
        if self._closed:
            return None
        try:
            return _as_rgb_frame(next(self._frames))
        except StopIteration:
            return None

    def close(self) -> None:
        self._closed = True


def create_screen_source(
    monitor: int = 1,
    *,
    fps: float = 30,
    region: tuple[int, int, int, int] | None = None,
    backend: str = "auto",
) -> FrameSource:
    """Create the preferred available screen source with a safe fallback."""

    selected = backend.casefold()
    if selected not in {"auto", "mss", "pillow"}:
        raise ValueError("Screen backend must be auto, mss, or pillow")
    if selected in {"auto", "mss"}:
        try:
            return MSSScreenSource(monitor, fps=fps, region=region)
        except CaptureBackendUnavailable:
            if selected == "mss":
                raise
    return PillowScreenSource(fps=min(fps, 15), region=region)


def create_frame_source(
    specification: str,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: float = 30,
    loop: bool = False,
    screen_backend: str = "auto",
) -> FrameSource:
    """Resolve ``camera:N``, ``screen:N``, or a video path."""

    kind, separator, value = specification.partition(":")
    normalized = kind.casefold()
    if separator and normalized in {"camera", "cam", "webcam"}:
        try:
            index = int(value)
        except ValueError as exc:
            raise ValueError("Camera source must look like camera:0") from exc
        return OpenCVFrameSource(index, width=width, height=height, fps=fps)
    if separator and normalized in {"screen", "desktop", "monitor"}:
        try:
            monitor = int(value)
        except ValueError as exc:
            raise ValueError("Screen source must look like screen:1") from exc
        return create_screen_source(monitor, fps=fps, backend=screen_backend)

    source = Path(specification).expanduser()
    if not source.is_file():
        raise CaptureError(
            f"Unknown live source {specification!r}; use camera:0, screen:1, "
            "or a video path"
        )
    return OpenCVFrameSource(source, fps=fps, loop=loop)


class LatestFramePump:
    """Capture on a daemon thread while retaining only the newest frame."""

    def __init__(self, source: FrameSource) -> None:
        self.source = source
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: CapturedFrame | None = None
        self._sequence = 0
        self._ended = False
        self._error: BaseException | None = None

    @property
    def ended(self) -> bool:
        with self._condition:
            return self._ended

    @property
    def captured_frames(self) -> int:
        with self._condition:
            return self._sequence

    def start(self) -> "LatestFramePump":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run,
            name=f"glyph-forge-capture:{self.source.name}",
            daemon=True,
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                pixels = self.source.read()
                if pixels is None:
                    break
                frame = _as_rgb_frame(pixels)
                with self._condition:
                    self._sequence += 1
                    self._latest = CapturedFrame(
                        pixels=frame,
                        sequence=self._sequence,
                        captured_at=time.monotonic(),
                    )
                    self._condition.notify_all()
        except BaseException as exc:
            if not self._stop.is_set():
                with self._condition:
                    self._error = exc
        finally:
            with self._condition:
                self._ended = True
                self._condition.notify_all()

    def next_frame(
        self,
        after_sequence: int = 0,
        timeout: float | None = None,
    ) -> CapturedFrame | None:
        """Wait for and return a frame newer than ``after_sequence``."""

        if self._thread is None:
            self.start()
        with self._condition:
            ready = self._condition.wait_for(
                lambda: (
                    (
                        self._latest is not None
                        and self._latest.sequence > after_sequence
                    )
                    or self._ended
                    or self._error is not None
                ),
                timeout=timeout,
            )
            if not ready:
                return None
            if self._latest is not None and self._latest.sequence > after_sequence:
                return self._latest
            if self._error is not None:
                if isinstance(self._error, CaptureError):
                    raise self._error
                raise CaptureError(
                    f"Capture source failed: {self._error}"
                ) from self._error
            return None

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self.source.close()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(0.0, timeout))

    def __enter__(self) -> "LatestFramePump":
        return self.start()

    def __exit__(self, *_args: object) -> None:
        self.stop()


__all__ = [
    "CaptureError",
    "CaptureBackendUnavailable",
    "CapturedFrame",
    "FrameSource",
    "IterableFrameSource",
    "LatestFramePump",
    "MSSScreenSource",
    "OpenCVFrameSource",
    "PillowScreenSource",
    "RGBFrame",
    "create_frame_source",
    "create_screen_source",
]

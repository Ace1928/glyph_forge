"""Memory-bounded video frame extraction."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image, ImageSequence, UnidentifiedImageError

from ..runtime import python_install_hint


def _load_opencv() -> Any | None:
    try:
        import cv2  # type: ignore[import-untyped]
    except (ImportError, OSError):
        return None
    return cv2


def iter_video_images(
    video_path: str | Path,
    max_frames: int | None = None,
) -> Iterator[Image.Image]:
    """Yield RGB video/GIF frames without retaining the whole stream in memory."""

    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames cannot be negative")
    if max_frames == 0:
        return
    source = Path(video_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Video does not exist: {source}")

    cv2 = _load_opencv()
    if cv2 is not None:
        capture = cv2.VideoCapture(str(source))
        if capture.isOpened():
            count = 0
            try:
                while max_frames is None or count < max_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yield Image.fromarray(rgb).copy()
                    count += 1
            finally:
                capture.release()
            if count:
                return
        else:
            capture.release()

    try:
        with Image.open(source) as image:
            for count, frame in enumerate(ImageSequence.Iterator(image)):
                if max_frames is not None and count >= max_frames:
                    break
                yield frame.convert("RGB").copy()
    except UnidentifiedImageError as exc:
        if cv2 is None:
            raise RuntimeError(
                f"This video needs OpenCV; {python_install_hint('media')}"
            ) from exc
        raise RuntimeError(f"No decoder could open video: {source}") from exc


def video_to_images(
    video_path: str | Path,
    max_frames: int | None = None,
) -> list[Image.Image]:
    """Return extracted frames as a list for backwards compatibility.

    New streaming callers should use :func:`iter_video_images` to keep memory
    usage bounded for long or high-resolution videos.
    """

    return list(iter_video_images(video_path, max_frames=max_frames))


__all__ = ["iter_video_images", "video_to_images"]

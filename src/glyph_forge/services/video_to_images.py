"""Video frame extraction service."""

from typing import Iterator, Optional

from PIL import Image, ImageSequence

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


def video_to_images(video_path: str, max_frames: Optional[int] = None) -> Iterator[Image.Image]:
    """Yield frames from a video or GIF file.

    Args:
        video_path: Path to the input video or GIF.
        max_frames: Optional limit on the number of frames returned.

    Returns:
        Iterator of ``PIL.Image`` objects representing the frames.
    """
    # Prefer PIL for GIFs to preserve all frames
    if str(video_path).lower().endswith(".gif"):
        img = Image.open(video_path)
        try:
            for count, frame in enumerate(ImageSequence.Iterator(img)):
                if max_frames is not None and count >= max_frames:
                    break
                yield frame.convert("RGB")
        finally:
            try:
                img.close()
            except Exception:
                pass
        return

    if cv2 is not None:
        cap = cv2.VideoCapture(video_path)
        try:
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or (max_frames is not None and count >= max_frames):
                    break
                yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                count += 1
        finally:
            cap.release()
        return

    img = Image.open(video_path)
    try:
        for count, frame in enumerate(ImageSequence.Iterator(img)):
            if max_frames is not None and count >= max_frames:
                break
            yield frame.convert("RGB")
    finally:
        try:
            img.close()
        except Exception:
            pass

"""Video frame extraction service."""

from typing import List, Optional
from PIL import Image, ImageSequence

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


def video_to_images(video_path: str, max_frames: Optional[int] = None) -> List[Image.Image]:
    """Extract frames from a video or GIF file.

    Args:
        video_path: Path to the input video or GIF.
        max_frames: Optional limit on the number of frames returned.

    Returns:
        List of ``PIL.Image`` objects representing the frames.
    """
    images: List[Image.Image] = []

    if cv2 is not None:
        cap = cv2.VideoCapture(video_path)
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames is not None and count >= max_frames):
                break
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            count += 1
        cap.release()
    else:
        img = Image.open(video_path)
        for count, frame in enumerate(ImageSequence.Iterator(img)):
            if max_frames is not None and count >= max_frames:
                break
            images.append(frame.convert("RGB"))

    return images

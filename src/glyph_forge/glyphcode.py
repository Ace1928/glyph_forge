"""Portable glyph codes: one base64 string that IS your artwork.

A glyph code is a self-contained, lossless snapshot of an image, a banner, or
an animated GIF, encoded as printable ASCII.  It needs no hosting, no upload,
and no server: paste the string anywhere (chat, mail, a note), and anyone can
regenerate the original artwork from it.

Formats (version 1):

- ``glyph:v1:img:<base64 PNG>``                — a lossless image snapshot
- ``glyph:v1:banner:<base64 JSON>``            — banner text + style settings
- ``glyph:v1:gif:<base64 JSON>;<frame>;...``   — animated frames + timing
"""

from __future__ import annotations

import base64
import binascii
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .persistence import atomic_write_bytes

_PREFIX = "glyph:v1:"
_TOTAL_BYTES_LIMIT = 8 * 1024 * 1024
_MAX_GIF_FRAMES = 96
_B64 = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")


class GlyphCodeError(ValueError):
    """Raised when a glyph code cannot be encoded or decoded."""


@dataclass(frozen=True, slots=True)
class BannerSpec:
    """The settings needed to regenerate one banner exactly."""

    text: str
    font: str = "small"
    style: str = "minimal"
    width: Optional[int] = None
    effects: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "text": self.text,
            "font": self.font,
            "style": self.style,
        }
        if self.width is not None:
            payload["width"] = self.width
        if self.effects is not None:
            payload["effects"] = self.effects
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "BannerSpec":
        try:
            text = str(payload.get("text", ""))
            if not text:
                raise GlyphCodeError("banner code is missing its text")
            width: Optional[int] = None
            if "width" in payload:
                width = int(str(payload["width"]))
            effects: Optional[str] = None
            if "effects" in payload:
                effects = str(payload["effects"])
            return cls(
                text=text,
                font=str(payload.get("font", "small")),
                style=str(payload.get("style", "minimal")),
                width=width,
                effects=effects,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GlyphCodeError(f"malformed banner code: {exc}") from exc


@dataclass(slots=True)
class DecodedGlyphCode:
    """The payload carried by one glyph code."""

    kind: str
    image: Optional[bytes] = None
    banner: Optional[BannerSpec] = None
    frames: List[bytes] = field(default_factory=list)
    frame_durations_ms: List[int] = field(default_factory=list)

    @property
    def is_animated(self) -> bool:
        return self.kind == "gif"

    def image_size(self) -> Optional[tuple[int, int]]:
        if self.image is None:
            return None
        from PIL import Image

        with Image.open(io.BytesIO(self.image)) as opened:
            return opened.size

    def banner_text(self) -> Optional[str]:
        if self.banner is None:
            return None
        from .services.text_to_banner import text_to_banner

        effects: Optional[List[str]] = None
        if self.banner.effects:
            effects = [
                item.strip() for item in self.banner.effects.split(",") if item.strip()
            ]
        return text_to_banner(
            self.banner.text,
            font=self.banner.font,
            style=self.banner.style,
            width=self.banner.width,
            effects=effects,
        )

    def save_image(self, path: Path) -> None:
        if self.image is None:
            raise GlyphCodeError("this code carries no image")
        atomic_write_bytes(_as_path(path), self.image)

    def save_gif(self, path: Path, loop: int = 0) -> None:
        if not self.frames:
            raise GlyphCodeError("this code carries no frames")
        fps = self.fps
        durations = self.frame_durations_ms or [int(1000 / max(1, fps))] * len(
            self.frames
        )
        from PIL import Image

        images = []
        for frame in self.frames:
            with Image.open(io.BytesIO(frame)) as opened:
                images.append(opened.convert("RGB"))
        first, rest = images[0], images[1:]
        stream = io.BytesIO()
        try:
            first.save(
                stream,
                format="GIF",
                save_all=True,
                append_images=rest,
                duration=durations,
                loop=loop,
                disposal=2,
            )
            atomic_write_bytes(_as_path(path), stream.getvalue())
        finally:
            for image in images:
                image.close()

    @property
    def fps(self) -> float:
        if not self.frame_durations_ms:
            return 12.0
        average = sum(self.frame_durations_ms) / len(self.frame_durations_ms)
        return round(1000.0 / average, 2)


def encode_image(image_path: Path, *, max_bytes: int = _TOTAL_BYTES_LIMIT) -> str:
    """Encode any still image as a lossless, byte-exact glyph code."""
    from PIL import Image

    path = _as_path(image_path)
    size = path.stat().st_size
    _check_size(size, max_bytes)
    try:
        with Image.open(str(path)) as opened:
            opened.verify()
    except OSError as exc:
        raise GlyphCodeError(f"could not read image: {exc}") from exc
    payload = path.read_bytes()
    return _compose("img", payload)


def encode_banner(
    text: str,
    *,
    font: str = "small",
    style: str = "minimal",
    width: Optional[int] = None,
    effects: Optional[str] = None,
) -> str:
    """Encode banner text plus its style settings as a glyph code."""
    spec = BannerSpec(text=text, font=font, style=style, width=width, effects=effects)
    payload = json.dumps(
        spec.to_dict(), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return _compose("banner", payload)


def encode_gif(
    gif_path: Path,
    *,
    max_frames: int = _MAX_GIF_FRAMES,
    max_bytes: int = _TOTAL_BYTES_LIMIT,
) -> str:
    """Encode an animated GIF as frames plus timing in one glyph code."""
    from PIL import Image

    frames: List[bytes] = []
    durations: List[int] = []
    try:
        with Image.open(str(gif_path)) as opened:
            while True:
                try:
                    frame = (
                        opened.convert("P", palette=Image.Palette.ADAPTIVE)
                        if opened.mode != "P"
                        else opened
                    )
                except OSError:
                    break
                buffer = io.BytesIO()
                frame.save(buffer, format="PNG")
                frames.append(buffer.getvalue())
                durations.append(int(opened.info.get("duration", 80)))
                if len(frames) >= max_frames or len(frames) >= getattr(
                    opened, "n_frames", 1
                ):
                    break
                opened.seek(len(frames))
    except OSError as exc:
        raise GlyphCodeError(f"could not read GIF: {exc}") from exc
    if len(frames) <= 1:
        return encode_image(gif_path, max_bytes=max_bytes)
    meta = json.dumps(
        {"durations": durations}, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    body_size = len(meta) + sum(len(frame) + 1 for frame in frames)
    _check_size(body_size, max_bytes)
    return (
        _PREFIX
        + "gif:"
        + base64.b64encode(meta).decode("ascii")
        + "~"
        + "~".join(base64.b64encode(frame).decode("ascii") for frame in frames)
    )


def encode_auto(path: Path, *, max_bytes: int = _TOTAL_BYTES_LIMIT) -> str:
    """Encode an image or animated GIF by sniffing the file content."""
    with _as_path(path).open("rb") as stream:
        head = stream.read(6)
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return encode_gif(path, max_bytes=max_bytes)
    return encode_image(path, max_bytes=max_bytes)


def decode_code(code: str) -> DecodedGlyphCode:
    """Decode a glyph code back into its snapshot payload."""
    if not code.startswith(_PREFIX):
        raise GlyphCodeError("not a glyph code (expected 'glyph:v1:…')")
    body = code[len(_PREFIX) :]
    kind, separator, payload = body.partition(":")
    if not separator:
        raise GlyphCodeError(f"malformed glyph code kind: {body!r}")
    if kind == "img":
        return DecodedGlyphCode(kind=kind, image=_decode_b64(payload))
    if kind == "banner":
        try:
            raw = json.loads(_decode_b64(payload).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GlyphCodeError(f"malformed banner code: {exc}") from exc
        if not isinstance(raw, dict):
            raise GlyphCodeError("malformed banner code: payload is not an object")
        return DecodedGlyphCode(kind=kind, banner=BannerSpec.from_dict(raw))
    if kind == "gif":
        meta_encoded, tilde, frames_part = payload.partition("~")
        if not tilde or not frames_part:
            raise GlyphCodeError("malformed GIF code: missing frames")
        try:
            meta = json.loads(_decode_b64(meta_encoded).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GlyphCodeError(f"malformed GIF code: {exc}") from exc
        raw_durations = meta.get("durations", []) if isinstance(meta, dict) else []
        durations = _sanitize_durations(raw_durations, frames_part)
        frames = [_decode_b64(part) for part in frames_part.split("~")]
        return DecodedGlyphCode(
            kind=kind,
            frames=frames,
            frame_durations_ms=durations,
        )
    raise GlyphCodeError(f"unknown glyph code kind {kind!r}")


def _compose(kind: str, payload: bytes) -> str:
    return _PREFIX + kind + ":" + base64.b64encode(payload).decode("ascii")


def _decode_b64(encoded: str) -> bytes:
    if not encoded or any(char not in _B64 for char in encoded):
        raise GlyphCodeError("glyph code contains invalid base64")
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise GlyphCodeError(f"glyph code contains invalid base64: {exc}") from exc


def _check_size(size: int, max_bytes: int) -> None:
    if size > max_bytes:
        raise GlyphCodeError(
            f"payload of {size} bytes exceeds the {max_bytes}-byte glyph code limit"
        )


def _sanitize_durations(raw: object, frames_part: str) -> List[int]:
    frame_count = len(frames_part.split("~"))
    if not isinstance(raw, list) or not raw:
        return [80] * frame_count
    durations: List[int] = []
    for value in raw[:frame_count]:
        try:
            durations.append(max(10, min(10_000, int(value))))
        except (TypeError, ValueError):
            durations.append(80)
    while len(durations) < frame_count:
        durations.append(80)
    return durations


def _as_path(path: Path) -> Path:
    return Path(path)


__all__ = [
    "BannerSpec",
    "DecodedGlyphCode",
    "GlyphCodeError",
    "decode_code",
    "encode_auto",
    "encode_banner",
    "encode_gif",
    "encode_image",
]

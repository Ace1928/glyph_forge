"""Sharing utilities for Glyph Forge CLI."""

from __future__ import annotations

import base64
import binascii
import gzip
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Tuple

from glyph_forge.exceptions import GlyphForgeInputError

from ..renderers import HTMLRenderer, SVGRenderer
from ..streaming.naming import derive_title, slugify

LINK_PREFIX = "glyphforge://"
MAX_SHARE_BYTES = 16 * 1024 * 1024
MAX_LINK_CHARACTERS = 32 * 1024 * 1024
FORMAT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,15}$")


def _validated_format(value: str) -> str:
    normalized = value.lower().strip()
    if not FORMAT_RE.fullmatch(normalized):
        raise GlyphForgeInputError(
            "share format must be 1-16 lowercase letters, numbers, dots, underscores, or hyphens"
        )
    return normalized


def _decode_base64(value: str, *, label: str) -> bytes:
    try:
        return base64.b64decode(value.encode("ascii"), altchars=b"-_", validate=True)
    except (UnicodeEncodeError, ValueError, binascii.Error) as exc:
        raise GlyphForgeInputError(f"share link contains invalid {label} Base64") from exc


def _bounded_gzip_decompress(data: bytes) -> bytes:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as archive:
            decoded = archive.read(MAX_SHARE_BYTES + 1)
    except (EOFError, OSError) as exc:
        raise GlyphForgeInputError("share link contains invalid gzip data") from exc
    if len(decoded) > MAX_SHARE_BYTES:
        raise GlyphForgeInputError(
            f"decoded share payload exceeds the {MAX_SHARE_BYTES}-byte safety limit"
        )
    return decoded


def build_share_path(
    source: str | None,
    fmt: str,
    output: str | None = None,
    title: str | None = None,
) -> Path:
    """Build a deterministic share output path."""
    if output:
        return Path(output)
    stem = slugify(derive_title(source or "glyph_forge", title))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path.cwd() / "glyph_forge_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_{timestamp}.{fmt}"


def render_share(content: str, fmt: str) -> str:
    """Render string content into a shareable text format."""
    if not isinstance(fmt, str):
        raise GlyphForgeInputError("share format must be text")
    if fmt == "txt":
        return str(content)
    if fmt == "html":
        if "<pre" in content or "<html" in content:
            return content
        matrix = _content_to_matrix(content)
        return HTMLRenderer().render(matrix)
    if fmt == "svg":
        matrix = _content_to_matrix(content)
        return SVGRenderer().render(matrix)
    raise GlyphForgeInputError(f"unsupported share format: {fmt}")


def write_share(content: str | Sequence[str], fmt: str, path: Path) -> None:
    """Write share output to file (supports binary formats)."""
    path = Path(path)
    if not isinstance(fmt, str):
        raise GlyphForgeInputError("share format must be text")
    fmt = fmt.lower()
    allowed = {"gif", "html", "png", "svg", "txt"}
    if fmt not in allowed:
        choices = ", ".join(sorted(allowed))
        raise GlyphForgeInputError(f"unsupported share format {fmt!r}; choose one of: {choices}")
    if not isinstance(content, str):
        if fmt != "gif":
            raise GlyphForgeInputError("frame sequences can only be written as GIF")
        _render_sequence_gif(content, path)
        return
    if fmt in {"txt", "html", "svg"}:
        atomic_write_text(path, render_share(content, fmt))
        return
    if fmt in {"png", "gif"}:
        if _contains_ansi(content):
            _render_ansi_image(content, path, fmt)
        else:
            _render_text_image(content, path, fmt)
        return


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes with flush, fsync, and same-directory atomic replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically write UTF-8 text."""

    atomic_write_bytes(path, content.encode("utf-8"))


def export_video_share(video_path: Path, fmt: str, output_path: Path) -> bool:
    """Export glyph video to shareable formats."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    if not video_path.exists():
        return False
    if fmt == "mp4":
        output_path.write_bytes(video_path.read_bytes())
        return True
    if fmt == "png":
        return _ffmpeg_extract_frame(video_path, output_path)
    if fmt == "gif":
        return _ffmpeg_extract_gif(video_path, output_path)
    if fmt == "apng":
        return _ffmpeg_extract_apng(video_path, output_path)
    if fmt == "webm":
        return _ffmpeg_extract_webm(video_path, output_path)
    return False


def try_open_path(path: Path) -> bool:
    """Best-effort open a file with system default application."""
    opener = None
    if shutil.which("xdg-open"):
        opener = ["xdg-open", str(path)]
    elif shutil.which("open"):
        opener = ["open", str(path)]
    if opener:
        subprocess.run(opener, check=False)
        return True
    return False


def copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard using available mechanisms."""
    try:
        import pyperclip  # type: ignore

        pyperclip.copy(text)
        return True
    except Exception:
        pass

    candidates: list[list[str]] = []
    if shutil.which("pbcopy"):
        candidates.append(["pbcopy"])
    if shutil.which("wl-copy"):
        candidates.append(["wl-copy"])
    if shutil.which("xclip"):
        candidates.append(["xclip", "-selection", "clipboard"])
    for cmd in candidates:
        try:
            subprocess.run(cmd, input=text.encode(), check=False)
            return True
        except Exception:
            continue
    return False


def encode_share_link(
    data: bytes,
    fmt: str,
    filename: str,
    source: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Encode data into a portable share link."""
    if not isinstance(data, bytes):
        raise GlyphForgeInputError("share data must be bytes")
    if len(data) > MAX_SHARE_BYTES:
        raise GlyphForgeInputError(f"share payload exceeds the {MAX_SHARE_BYTES}-byte safety limit")
    fmt = _validated_format(fmt)
    filename = Path(filename.replace("\\", "/")).name
    if not filename or filename in {".", ".."}:
        raise GlyphForgeInputError("share filename must name a file")
    compression = None
    compressed = data
    if len(data) > 2048:
        trial = gzip.compress(data)
        if len(trial) < len(data):
            compressed = trial
            compression = "gzip"

    payload = {
        "version": 1,
        "format": fmt,
        "filename": filename,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "compression": compression,
        "data_b64": base64.urlsafe_b64encode(compressed).decode("ascii"),
    }
    if metadata:
        payload["metadata"] = metadata
    try:
        blob = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GlyphForgeInputError("share metadata must contain finite JSON data") from exc
    link = LINK_PREFIX + base64.urlsafe_b64encode(blob).decode("ascii")
    if len(link) > MAX_LINK_CHARACTERS:
        raise GlyphForgeInputError("encoded share link exceeds the safety limit")
    return link


def decode_share_link(link: str) -> Tuple[dict, bytes]:
    """Decode a share link back into payload and raw bytes."""
    if not isinstance(link, str):
        raise GlyphForgeInputError("share link must be text")
    raw = link.strip()
    if raw.startswith(LINK_PREFIX):
        raw = raw[len(LINK_PREFIX) :]
    if not raw or len(raw) > MAX_LINK_CHARACTERS:
        raise GlyphForgeInputError("share link is empty or exceeds the safety limit")
    blob = _decode_base64(raw, label="envelope")
    try:
        payload = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GlyphForgeInputError("share link envelope is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise GlyphForgeInputError("share link uses an unsupported envelope version")
    fmt = payload.get("format")
    filename = payload.get("filename")
    encoded_data = payload.get("data_b64")
    compression = payload.get("compression")
    if not isinstance(fmt, str):
        raise GlyphForgeInputError("share link format is missing")
    payload["format"] = _validated_format(fmt)
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
    ):
        raise GlyphForgeInputError("share link filename is invalid")
    if not isinstance(encoded_data, str):
        raise GlyphForgeInputError("share link data is missing")
    if compression not in {None, "gzip"}:
        raise GlyphForgeInputError("share link compression is unsupported")
    data = _decode_base64(encoded_data, label="payload")
    if compression == "gzip":
        data = _bounded_gzip_decompress(data)
    elif len(data) > MAX_SHARE_BYTES:
        raise GlyphForgeInputError(
            f"decoded share payload exceeds the {MAX_SHARE_BYTES}-byte safety limit"
        )
    return payload, data


def load_share_link(value: str) -> Tuple[dict, bytes]:
    """Load a share link from a raw string or a file path."""
    if value.startswith(LINK_PREFIX):
        return decode_share_link(value)
    try:
        path = Path(value)
        if path.is_file():
            if path.stat().st_size > MAX_LINK_CHARACTERS:
                raise GlyphForgeInputError("share link file exceeds the safety limit")
            return decode_share_link(path.read_text(encoding="utf-8").strip())
    except OSError as exc:
        raise GlyphForgeInputError(f"could not read share link file: {exc}") from exc
    return decode_share_link(value)


def _content_to_matrix(content: str) -> list[list[str]]:
    if not isinstance(content, str):
        content = str(content)
    lines = content.splitlines() or [content]
    return [list(line) for line in lines]


def _render_text_image(content: str, path: Path, fmt: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not isinstance(content, str):
        content = str(content)
    lines = content.splitlines() or [content]
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C:\\Windows\\Fonts\\consola.ttf",
    ]
    font = None
    for fp in font_paths:
        if Path(fp).exists():
            font = ImageFont.truetype(fp, 14)
            break
    if font is None:
        font = ImageFont.load_default()

    max_width = max(len(line) for line in lines)
    char_w, char_h = font.getbbox("M")[2:4]
    img_w = max(1, max_width * char_w)
    img_h = max(1, len(lines) * (char_h + 2))

    img = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = 0
    for line in lines:
        draw.text((0, y), line, font=font, fill=(255, 255, 255))
        y += char_h + 2

    if fmt == "png":
        img.save(path, format="PNG")
    elif fmt == "gif":
        img.save(path, format="GIF", save_all=True, duration=100, loop=0)


def _render_ansi_image(content: str, path: Path, fmt: str) -> None:
    import tempfile
    import uuid

    from PIL import Image

    from glyph_forge.streaming.core.recorder import GlyphRecorder, RecorderConfig

    temp_path = Path(tempfile.gettempdir()) / f"glyph_render_{uuid.uuid4().hex}.mp4"
    try:
        recorder = GlyphRecorder(RecorderConfig(output_path=temp_path))
        frame = recorder.render_to_image(content)
        img = Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB
    except Exception:
        # Keep share export functional when optional video backends are unavailable.
        _render_text_image(content, path, fmt)
        return
    if fmt == "png":
        img.save(path, format="PNG")
    elif fmt == "gif":
        img.save(path, format="GIF", save_all=True, duration=100, loop=0)


def _render_sequence_gif(frames: Sequence[str], path: Path, duration_ms: int = 80) -> None:
    from PIL import Image

    frames_list = list(frames)
    if not frames_list:
        return
    if any(_contains_ansi(frame) for frame in frames_list):
        imgs = [_ansi_frame_to_image(frame) for frame in frames_list]
    else:
        imgs = [_text_frame_to_image(frame) for frame in frames_list]
    imgs = [img.convert("P", palette=Image.ADAPTIVE) for img in imgs]
    imgs[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def _contains_ansi(content: str) -> bool:
    if not isinstance(content, str):
        return False
    return "\033[" in content


def _text_frame_to_image(content: str):
    from PIL import Image, ImageDraw, ImageFont

    if not isinstance(content, str):
        content = str(content)
    lines = content.splitlines() or [content]
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C:\\Windows\\Fonts\\consola.ttf",
    ]
    font = None
    for fp in font_paths:
        if Path(fp).exists():
            font = ImageFont.truetype(fp, 14)
            break
    if font is None:
        font = ImageFont.load_default()

    max_width = max(len(line) for line in lines)
    char_w, char_h = font.getbbox("M")[2:4]
    img_w = max(1, max_width * char_w)
    img_h = max(1, len(lines) * (char_h + 2))

    img = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = 0
    for line in lines:
        draw.text((0, y), line, font=font, fill=(255, 255, 255))
        y += char_h + 2
    return img


def _ansi_frame_to_image(content: str):
    import tempfile
    import uuid

    from PIL import Image

    from glyph_forge.streaming.core.recorder import GlyphRecorder, RecorderConfig

    temp_path = Path(tempfile.gettempdir()) / f"glyph_render_{uuid.uuid4().hex}.mp4"
    try:
        recorder = GlyphRecorder(RecorderConfig(output_path=temp_path))
        frame = recorder.render_to_image(content)
        return Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB
    except Exception:
        return _text_frame_to_image(content)


def _ffmpeg_extract_frame(video_path: Path, output_path: Path) -> bool:
    import subprocess

    if not shutil.which("ffmpeg"):
        return False
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and output_path.exists()


def _ffmpeg_extract_gif(video_path: Path, output_path: Path) -> bool:
    import subprocess

    if not shutil.which("ffmpeg"):
        return False
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "fps=12,scale=640:-1:flags=lanczos",
        "-loop",
        "0",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and output_path.exists()


def _ffmpeg_extract_apng(video_path: Path, output_path: Path) -> bool:
    import subprocess

    if not shutil.which("ffmpeg"):
        return False
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "fps=12,scale=640:-1:flags=lanczos",
        "-plays",
        "0",
        "-f",
        "apng",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and output_path.exists()


def _ffmpeg_extract_webm(video_path: Path, output_path: Path) -> bool:
    import subprocess

    if not shutil.which("ffmpeg"):
        return False
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libvpx-vp9",
        "-b:v",
        "2M",
        "-an",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and output_path.exists()

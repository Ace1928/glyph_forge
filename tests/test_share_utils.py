from pathlib import Path

import pytest

from glyph_forge.cli import share_utils
from glyph_forge.cli.share_utils import write_share
from glyph_forge.exceptions import GlyphForgeInputError


def test_write_share_sequence_gif(tmp_path: Path) -> None:
    frames = ["ABC\nDEF", "123\n456"]
    out = tmp_path / "sequence.gif"
    write_share(frames, "gif", out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_export_video_share_mp4(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    out = tmp_path / "out.mp4"

    assert share_utils.export_video_share(video, "mp4", out) is True
    assert out.read_bytes() == b"video"


def test_export_video_share_png(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    out = tmp_path / "out.png"
    called = {"value": False}

    def fake_extract(*_args, **_kwargs):
        called["value"] = True
        out.write_bytes(b"png")
        return True

    monkeypatch.setattr(share_utils, "_ffmpeg_extract_frame", fake_extract)

    assert share_utils.export_video_share(video, "png", out) is True
    assert called["value"] is True


def test_unknown_share_format_is_not_silently_written(tmp_path: Path) -> None:
    with pytest.raises(GlyphForgeInputError, match="unsupported share format"):
        write_share("content", "exe", tmp_path / "payload.exe")

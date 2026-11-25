"""Tests for audio muxing utilities."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from glyph_forge.streaming import audio_tools


def test_mux_audio_local(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    audio = tmp_path / "audio.m4a"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")

    monkeypatch.setattr(audio_tools.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    def fake_run(cmd, capture_output=True, text=True):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"muxed")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(audio_tools.subprocess, "run", fake_run)

    out = audio_tools.mux_audio(video, str(audio), overwrite=True)
    assert out is not None and out.exists()


def test_mux_audio_youtube(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    temp_audio = tmp_path / "yt_audio.m4a"
    temp_audio.write_bytes(b"audio")

    monkeypatch.setattr(audio_tools.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(audio_tools, "download_audio", lambda *_: temp_audio)

    def fake_run(cmd, capture_output=True, text=True):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"muxed")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(audio_tools.subprocess, "run", fake_run)

    out = audio_tools.mux_audio(video, "https://youtube.com/watch?v=demo", overwrite=True)
    assert out is not None and out.exists()


def test_mux_audio_requires_ffmpeg(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")

    monkeypatch.setattr(audio_tools.shutil, "which", lambda *_: None)
    with pytest.raises(RuntimeError):
        audio_tools.mux_audio(video, "audio.m4a")

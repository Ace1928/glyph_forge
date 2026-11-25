"""Tests for streaming extractors."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from glyph_forge.streaming import extractors
from glyph_forge.streaming.extractors import ExtractionResult, VideoSourceExtractor


def test_extract_playlist_arguments(monkeypatch) -> None:
    called = {}

    def fake_playlist(
        cls,
        url,
        resolution,
        include_audio,
        playlist_max_items=None,
        playlist_start=None,
        playlist_end=None,
        yt_cookies=None,
        yt_cookies_from_browser=None,
        yt_user_agent=None,
        yt_proxy=None,
        yt_skip_authcheck=None,
        yt_player_client=None,
    ):
        called["url"] = url
        called["resolution"] = resolution
        called["include_audio"] = include_audio
        called["playlist_max_items"] = playlist_max_items
        called["playlist_start"] = playlist_start
        called["playlist_end"] = playlist_end
        called["yt_cookies"] = yt_cookies
        called["yt_cookies_from_browser"] = yt_cookies_from_browser
        called["yt_user_agent"] = yt_user_agent
        called["yt_proxy"] = yt_proxy
        called["yt_skip_authcheck"] = yt_skip_authcheck
        called["yt_player_client"] = yt_player_client
        return ExtractionResult(video_url="file.mp4", title="playlist")

    monkeypatch.setattr(
        VideoSourceExtractor, "_extract_youtube_playlist", classmethod(fake_playlist)
    )

    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123"
    result = VideoSourceExtractor.extract(
        url,
        resolution=480,
        include_audio=True,
        playlist_max_items=3,
        playlist_start=2,
        playlist_end=4,
        yt_cookies="/tmp/cookies.txt",
        yt_cookies_from_browser="chrome",
        yt_user_agent="UA",
        yt_proxy="http://proxy",
        yt_skip_authcheck=True,
        yt_player_client="android,web",
    )

    assert isinstance(result, ExtractionResult)
    assert called["url"] == url
    assert called["resolution"] == 480
    assert called["include_audio"] is True
    assert called["playlist_max_items"] == 3
    assert called["playlist_start"] == 2
    assert called["playlist_end"] == 4
    assert called["yt_cookies"] == "/tmp/cookies.txt"
    assert called["yt_cookies_from_browser"] == "chrome"
    assert called["yt_user_agent"] == "UA"
    assert called["yt_proxy"] == "http://proxy"
    assert called["yt_skip_authcheck"] is True
    assert called["yt_player_client"] == "android,web"


def test_extract_playlist_stitches_audio(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(extractors, "HAS_YT_DLP", True)
    monkeypatch.setattr(
        extractors.shutil, "which", lambda binary: "/usr/bin/ffmpeg" if binary == "ffmpeg" else None
    )
    monkeypatch.setattr(extractors.tempfile, "gettempdir", lambda: str(tmp_path))

    class DummyYDL:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def extract_info(self, url, download=False):
            return {"title": "Demo Playlist"}

    monkeypatch.setattr(extractors, "yt_dlp", SimpleNamespace(YoutubeDL=DummyYDL))

    def fake_run(cmd, capture_output=False, text=False):
        class Result:
            def __init__(self):
                self.returncode = 0
                self.stderr = ""

        if cmd and cmd[0] == "yt-dlp":
            out_pattern = Path(cmd[cmd.index("-o") + 1])
            out_dir = out_pattern.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "001_demo.mp4").write_bytes(b"video")
            (out_dir / "002_demo.mp4").write_bytes(b"video")
        elif cmd and cmd[0] == "ffmpeg":
            out_path = Path(cmd[-1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"data")
        return Result()

    monkeypatch.setattr(extractors.subprocess, "run", fake_run)

    def fake_extract_local(cls, path, include_audio=True):
        return extractors.ExtractionResult(video_url=path, audio_url=path, title="local")

    monkeypatch.setattr(
        extractors.VideoSourceExtractor,
        "_extract_local_file",
        classmethod(fake_extract_local),
    )

    result = extractors.VideoSourceExtractor._extract_youtube_playlist(
        "https://www.youtube.com/watch?v=demo&list=PL123",
        resolution=720,
        include_audio=True,
    )

    assert Path(result.video_url).name == "playlist_merged.mp4"
    assert Path(result.audio_url).name == "playlist_audio.m4a"
    assert result.title == "Demo Playlist"
    assert result.format == "playlist"


def test_parse_cookies_from_browser_spec() -> None:
    parsed = extractors._parse_cookies_from_browser_spec("firefox:profile:container")
    assert parsed == ("firefox", "profile", None, "container")
    parsed = extractors._parse_cookies_from_browser_spec("firefox+kwallet")
    assert parsed == ("firefox", None, "kwallet", None)


def test_parse_player_client_spec() -> None:
    assert extractors._parse_player_client_spec(None) is None
    assert extractors._parse_player_client_spec("") is None
    assert extractors._parse_player_client_spec("android,web") == ["android", "web"]

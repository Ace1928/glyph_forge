"""Optional network-media resolver tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from glyph_forge.live import capture, network


class FakeYoutubeDL:
    options: dict[str, object] = {}

    def __init__(self, options: dict[str, object]) -> None:
        type(self).options = options

    def __enter__(self) -> "FakeYoutubeDL":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def extract_info(self, url: str, download: bool) -> dict[str, object]:
        assert download is False
        return {
            "title": "Tiny stream",
            "webpage_url": url,
            "duration": 12.5,
            "requested_formats": [
                {"url": "https://media.invalid/audio", "vcodec": "none"},
                {
                    "url": "https://media.invalid/video",
                    "vcodec": "vp9",
                    "width": 640,
                    "height": 360,
                    "fps": 30,
                },
            ],
        }


def test_resolver_selects_video_stream_without_downloading(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))

    result = network.resolve_network_source(
        "https://videos.invalid/watch/123",
        max_height=360,
    )

    assert result.url == "https://media.invalid/video"
    assert result.title == "Tiny stream"
    assert result.height == 360
    assert "height<=?360" in str(FakeYoutubeDL.options["format"])
    assert FakeYoutubeDL.options["skip_download"] is True


def test_missing_network_extra_has_actionable_error(monkeypatch) -> None:
    real_import = network.importlib.import_module

    def missing(name: str):
        if name == "yt_dlp":
            raise ImportError(name)
        return real_import(name)

    monkeypatch.setattr(network.importlib, "import_module", missing)

    with pytest.raises(network.NetworkSourceError, match=r"glyphforge\[network\]"):
        network.resolve_network_source("https://videos.invalid/watch/123")


def test_capture_factory_accepts_url_prefix(monkeypatch) -> None:
    resolved = network.ResolvedNetworkSource(
        url="https://media.invalid/video",
        title="Remote title",
        webpage_url="https://videos.invalid/watch/123",
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(network, "resolve_network_source", lambda *_a, **_k: resolved)

    def fake_source(source, **kwargs):
        calls.update(source=source, **kwargs)
        return SimpleNamespace(name=kwargs["label"])

    monkeypatch.setattr(capture, "OpenCVFrameSource", fake_source)
    result = capture.create_frame_source(
        "url:https://videos.invalid/watch/123",
        height=480,
        fps=24,
    )

    assert result.name == "Remote title"
    assert calls["source"] == "https://media.invalid/video"
    assert calls["fps"] == 24
    assert calls["loop"] is False

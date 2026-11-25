"""Targeted CLI tests for skipped auto-coverage functions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_cli_interactive_noop(monkeypatch):
    pytest.importorskip("textual", reason="textual not installed")
    import glyph_forge.cli as cli
    from glyph_forge.ui import tui

    class DummyApp:
        def run(self):
            return None

    monkeypatch.setattr(tui, "GlyphForgeApp", DummyApp)
    cli.interactive()


def test_cli_stream_screen_capture(monkeypatch):
    import glyph_forge.cli as cli

    class DummyFFmpeg:
        def __init__(self, *args, **kwargs):
            self._started = False

        def start_recording(self, duration=None):
            self._started = True
            return True

        def stop_recording(self):
            return None

    class DummyFirefox:
        def launch(self):
            return True

        def navigate(self, url):
            return True

        def fullscreen(self):
            return True

    monkeypatch.setattr(cli, "FFmpegCapture", DummyFFmpeg, raising=False)
    monkeypatch.setattr(cli, "FirefoxController", DummyFirefox, raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda *_args, **_kwargs: "/usr/bin/ffmpeg")

    # Make time advance in a controlled way to exercise the loop once
    times = iter([0.0, 0.0, 0.6, 1.2])
    monkeypatch.setattr(cli.time, "time", lambda: next(times))
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)

    cli._stream_screen_capture(
        url=None, duration=1, mode="gradient", color="ansi256", record="none", stats=True
    )


def test_cli_stream_engine_noop(monkeypatch):
    from typer.testing import CliRunner

    import glyph_forge.cli as cli
    import glyph_forge.streaming.engine as engine_module

    class DummyEngine:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            return None

    monkeypatch.setattr(engine_module, "UnifiedStreamEngine", DummyEngine)
    monkeypatch.setattr(
        engine_module,
        "UnifiedStreamConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    result = CliRunner().invoke(cli.app, ["stream", "video.mp4", "--record", "none"])

    assert result.exit_code == 0, result.output

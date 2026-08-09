"""Tests for portable capture and bounded-latency live sessions."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import typer
from rich.text import Text
from typer.testing import CliRunner

from glyph_forge.cli import app
from glyph_forge.cli import live as live_cli
from glyph_forge.live import capture
from glyph_forge.live.capture import (
    CaptureBackendUnavailable,
    CaptureError,
    IterableFrameSource,
    LatestFramePump,
    MSSScreenSource,
    OpenCVFrameSource,
    create_frame_source,
    create_screen_source,
)
from glyph_forge.live.renderers import (
    FrameRenderer,
    RenderConfig,
    RenderMode,
    RenderResult,
)
from glyph_forge.live.session import (
    TerminalPresenter,
    TerminalSessionConfig,
    run_terminal_session,
)


def test_iterable_source_normalizes_gray_and_rgba_frames() -> None:
    source = IterableFrameSource(
        [
            np.full((2, 3), 12, dtype=np.uint8),
            np.full((2, 3, 4), 24, dtype=np.uint8),
        ]
    )

    gray = source.read()
    rgba = source.read()

    assert gray is not None and gray.shape == (2, 3, 3)
    assert rgba is not None and rgba.shape == (2, 3, 3)
    assert source.read() is None


def test_latest_frame_pump_drops_stale_burst_frames() -> None:
    frames = [np.full((1, 1, 3), value, dtype=np.uint8) for value in range(5)]
    pump = LatestFramePump(IterableFrameSource(frames)).start()

    # Waiting for an impossible sequence lets the finite producer reach EOF.
    assert pump.next_frame(999, timeout=1) is None
    latest = pump.next_frame(0, timeout=0)

    assert latest is not None
    assert latest.sequence == 5
    assert int(latest.pixels[0, 0, 0]) == 4
    assert pump.captured_frames == 5
    pump.stop()


class _BrokenSource:
    name = "broken"

    def read(self) -> np.ndarray:
        raise RuntimeError("camera disconnected")

    def close(self) -> None:
        pass


def test_capture_thread_errors_surface_to_the_consumer() -> None:
    pump = LatestFramePump(_BrokenSource()).start()

    with pytest.raises(CaptureError, match="camera disconnected"):
        pump.next_frame(timeout=1)
    pump.stop()


def test_terminal_session_renders_and_returns_metrics() -> None:
    source = IterableFrameSource(
        [np.full((4, 2, 3), 255, dtype=np.uint8)],
        name="one-frame",
    )
    renderer = FrameRenderer(
        RenderConfig(width=1, height=1, mode="braille", color="none")
    )
    output = io.StringIO()

    stats = run_terminal_session(
        source,
        renderer,
        TerminalSessionConfig(
            target_fps=120,
            max_frames=1,
            alternate_screen=False,
            show_stats=False,
        ),
        output=output,
    )

    assert output.getvalue() == "⣿\n"
    assert stats.source == "one-frame"
    assert stats.captured_frames == 1
    assert stats.presented_frames == 1
    assert stats.dropped_frames == 0
    assert stats.output_bytes == len("⣿\n".encode())
    assert stats.full_redraws == 1


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_terminal_session_restores_the_terminal_surface() -> None:
    output = _TTYBuffer()
    renderer = FrameRenderer(RenderConfig(width=1, height=1, charset="@"))

    run_terminal_session(
        IterableFrameSource([np.zeros((1, 1, 3), dtype=np.uint8)]),
        renderer,
        TerminalSessionConfig(max_frames=1, show_stats=False),
        output=output,
    )

    text = output.getvalue()
    assert text.startswith("\x1b[?1049h\x1b[?25l")
    assert text.endswith("\x1b[0m\x1b[?25h\x1b[?1049l")


def test_terminal_presenter_emits_exact_changed_row_delta() -> None:
    output = _TTYBuffer()
    presenter = TerminalPresenter(output, alternate_screen=True, redraw="delta")
    first = RenderResult("aa\nbb\ncc", 2, 3, RenderMode.GLYPH)
    second = RenderResult("aa\nbc\ncc", 2, 3, RenderMode.GLYPH)

    presenter.present(first)
    boundary = len(output.getvalue())
    presenter.present(second)

    assert output.getvalue()[boundary:] == "\x1b[2;1Hbc\x1b[K"
    assert presenter.full_redraws == 1
    assert presenter.delta_redraws == 1


def test_terminal_presenter_auto_selects_the_smaller_update() -> None:
    output = _TTYBuffer()
    presenter = TerminalPresenter(output, alternate_screen=True)
    row = "a" * 100
    first = RenderResult("\n".join([row] * 4), 100, 4, RenderMode.GLYPH)
    changed = "b" + row[1:]
    sparse = RenderResult(
        "\n".join([row, changed, row, row]),
        100,
        4,
        RenderMode.GLYPH,
    )
    busy = RenderResult("\n".join(["c" * 100] * 4), 100, 4, RenderMode.GLYPH)

    presenter.present(first)
    presenter.present(sparse)
    presenter.present(busy)

    assert presenter.full_redraws == 2
    assert presenter.delta_redraws == 1


def test_terminal_presenter_skips_unchanged_surface() -> None:
    output = _TTYBuffer()
    presenter = TerminalPresenter(output, alternate_screen=True)
    result = RenderResult("static", 6, 1, RenderMode.GLYPH)

    presenter.present(result)
    boundary = len(output.getvalue())
    presenter.present(result)

    assert output.getvalue()[boundary:] == ""
    assert presenter.skipped_redraws == 1


def test_terminal_presenter_falls_back_to_full_redraw_after_resize() -> None:
    output = _TTYBuffer()
    presenter = TerminalPresenter(output, alternate_screen=True, redraw="delta")
    presenter.present(RenderResult("old", 3, 1, RenderMode.GLYPH))
    boundary = len(output.getvalue())

    presenter.present(RenderResult("new\nsize", 4, 2, RenderMode.GLYPH))

    assert output.getvalue()[boundary:] == "\x1b[Hnew\nsize\x1b[J"
    assert presenter.full_redraws == 2
    assert presenter.delta_redraws == 0


def test_adaptive_presentation_reduces_static_terminal_bytes() -> None:
    result = RenderResult("\n".join(["x" * 100] * 10), 100, 10, RenderMode.GLYPH)

    def write_sequence(redraw: str) -> int:
        presenter = TerminalPresenter(
            _TTYBuffer(),
            alternate_screen=True,
            redraw=redraw,
        )
        for frame in range(20):
            presenter.present(result, f"frame {frame}")
        presenter.close()
        return presenter.output_bytes

    assert write_sequence("auto") < write_sequence("full") // 5


@pytest.mark.parametrize(
    "config",
    [
        TerminalSessionConfig(target_fps=0),
        TerminalSessionConfig(duration=0),
        TerminalSessionConfig(max_frames=0),
        TerminalSessionConfig(redraw="paint"),
    ],
)
def test_terminal_session_rejects_invalid_limits(config: TerminalSessionConfig) -> None:
    with pytest.raises(ValueError):
        config.validated()


class _FakeCapture:
    def __init__(self) -> None:
        self.released = 0
        self.settings: list[tuple[int, float]] = []
        self.frames = iter([np.asarray([[[1, 2, 3]]], dtype=np.uint8)])

    def isOpened(self) -> bool:
        return True

    def set(self, property_id: int, value: float) -> bool:
        self.settings.append((property_id, value))
        return True

    def get(self, _property_id: int) -> float:
        return 30.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self) -> None:
        self.released += 1


class _FakeCv2:
    CAP_PROP_FRAME_WIDTH = 1
    CAP_PROP_FRAME_HEIGHT = 2
    CAP_PROP_FPS = 3
    CAP_PROP_POS_FRAMES = 4

    def __init__(self, device: _FakeCapture) -> None:
        self.device = device

    def VideoCapture(self, _source: int | str) -> _FakeCapture:
        return self.device


def test_opencv_camera_sets_requested_properties_and_converts_bgr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _FakeCapture()
    monkeypatch.setattr(capture, "_load_opencv", lambda: _FakeCv2(device))
    source = OpenCVFrameSource(0, width=640, height=480, fps=24)

    frame = source.read()
    source.close()
    source.close()

    assert frame is not None
    assert frame.tolist() == [[[3, 2, 1]]]
    assert device.settings == [(1, 640), (2, 480), (3, 24)]
    assert device.released == 1


def test_screen_factory_falls_back_only_when_backend_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = IterableFrameSource([])

    def unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise CaptureBackendUnavailable("not installed")

    monkeypatch.setattr(capture, "MSSScreenSource", unavailable)
    monkeypatch.setattr(capture, "PillowScreenSource", lambda **_kwargs: fallback)

    assert create_screen_source(backend="auto") is fallback
    with pytest.raises(CaptureBackendUnavailable):
        create_screen_source(backend="mss")
    with pytest.raises(CaptureBackendUnavailable, match="specific X11 display"):
        create_screen_source(backend="auto", display_name=":42")


def test_mss_screen_source_targets_an_explicit_display(monkeypatch) -> None:
    options: list[dict[str, str]] = []

    class FakeMSS:
        monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
            {"left": -1280, "top": 40, "width": 1280, "height": 720},
        ]

        def close(self) -> None:
            pass

    def factory(**kwargs):
        options.append(kwargs)
        return FakeMSS()

    monkeypatch.setitem(sys.modules, "mss", SimpleNamespace(MSS=factory))
    source = MSSScreenSource(1, display_name=":42")

    assert options == [{"display": ":42"}]
    assert source.name == ":42/screen:1"
    assert source.capture_region == capture.CaptureRegion(-1280, 40, 1280, 720)
    source.close()


@pytest.mark.parametrize("specification", ["camera:nope", "screen:nope", "missing.mov"])
def test_frame_source_specs_fail_clearly(specification: str) -> None:
    with pytest.raises((CaptureError, ValueError)):
        create_frame_source(specification)


def test_webcam_and_desktop_are_direct_unified_cli_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_run(specification: str, **options: Any) -> None:
        calls.append((specification, options))

    monkeypatch.setattr(live_cli, "_run_source", fake_run)
    runner = CliRunner()

    webcam = runner.invoke(app, ["webcam", "2", "--frames", "1"])
    desktop = runner.invoke(app, ["desktop", "0", "--frames", "1"])

    assert webcam.exit_code == 0, webcam.output
    assert desktop.exit_code == 0, desktop.output
    assert calls[0][0] == "camera:2"
    assert calls[1][0] == "screen:0"
    assert calls[0][1]["max_frames"] == 1


def test_live_commands_expose_adaptive_redraw_control() -> None:
    result = CliRunner().invoke(app, ["live", "screen", "--help"])
    plain_output = Text.from_ansi(result.output).plain

    assert result.exit_code == 0, result.output
    assert "--redraw" in plain_output
    assert "Terminal updates" in plain_output


def test_host_desktop_control_refuses_same_terminal_feedback() -> None:
    with pytest.raises(typer.Exit) as error:
        live_cli._run_source(
            "screen:0",
            mode="glyph",
            color="none",
            width=10,
            height=5,
            charset="general",
            invert=False,
            dither=False,
            edge_algorithm="sobel",
            edge_threshold=48,
            fps=1,
            duration=0.1,
            max_frames=1,
            performance="eco",
            control=True,
        )
    assert error.value.exit_code == 2


def test_live_video_command_requires_an_existing_file(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["live", "video", str(tmp_path / "missing.mp4")])

    assert result.exit_code != 0

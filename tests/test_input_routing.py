"""Input-routing tests with no real keyboard, pointer, or display access."""

from __future__ import annotations

import io
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from glyph_forge.live import input as input_module
from glyph_forge.live import session as session_module
from glyph_forge.live.capture import CaptureRegion, IterableFrameSource
from glyph_forge.live.input import (
    InputRouter,
    KeyInput,
    NullInputSink,
    PointerInput,
    RenderViewport,
    create_input_sink,
    map_pointer_to_capture,
)
from glyph_forge.live.renderers import FrameRenderer, RenderConfig
from glyph_forge.live.session import TerminalSessionConfig, run_terminal_session
from glyph_forge.live.terminal_input import (
    EmergencyInput,
    TerminalEventParser,
    TerminalInputPump,
)


class FakeSink:
    name = "fake"

    def __init__(self) -> None:
        self.keys: list[KeyInput] = []
        self.pointers: list[tuple[PointerInput, int, int]] = []
        self.releases = 0
        self.closes = 0

    def send_key(self, event: KeyInput) -> bool:
        self.keys.append(event)
        return True

    def send_pointer(self, event: PointerInput, x: int, y: int) -> bool:
        self.pointers.append((event, x, y))
        return True

    def release_all(self) -> None:
        self.releases += 1

    def close(self) -> None:
        self.closes += 1


def test_pointer_mapping_honors_viewport_offset_and_capture_origin() -> None:
    viewport = RenderViewport(columns=4, rows=2, left=3, top=2)
    capture = CaptureRegion(left=-100, top=50, width=400, height=200)

    assert map_pointer_to_capture(3, 2, viewport, capture) == (-50, 100)
    assert map_pointer_to_capture(6, 3, viewport, capture) == (250, 200)
    assert map_pointer_to_capture(2, 2, viewport, capture) is None
    assert map_pointer_to_capture(3, 4, viewport, capture) is None


def test_router_ignores_unmapped_pointer_but_routes_keys_immediately() -> None:
    sink = FakeSink()
    router = InputRouter(sink, CaptureRegion(10, 20, 100, 50))

    assert router.route(KeyInput("a"))
    assert not router.route(PointerInput("press", 1, 1, button="left"))
    router.update_viewport(10, 5)
    assert router.route(PointerInput("press", 1, 1, button="left"))
    assert not router.route(PointerInput("press", 11, 1, button="left"))

    assert sink.keys == [KeyInput("a")]
    assert sink.pointers[0][1:] == (15, 25)
    assert router.routed_events == 2
    router.close()
    router.close()
    assert sink.releases == 1
    assert sink.closes == 1


def test_terminal_parser_handles_fragmented_unicode_keys_and_hard_escape() -> None:
    parser = TerminalEventParser()
    glyph = "界".encode()

    assert parser.feed(glyph[:1]) == []
    assert parser.feed(glyph[1:]) == [KeyInput("界")]
    assert parser.feed(b"\x03") == [KeyInput("c", frozenset({"ctrl"}))]
    assert parser.feed(b"\x1bx") == [KeyInput("x", frozenset({"alt"}))]
    assert parser.feed(b"\x1d") == [EmergencyInput()]
    assert parser.feed(b"\x1b") == []
    assert parser.flush() == [KeyInput("escape")]


def test_terminal_parser_handles_navigation_modifiers_and_fragmented_mouse() -> None:
    parser = TerminalEventParser()

    assert parser.feed(b"\x1b[1;6A") == [KeyInput("up", frozenset({"shift", "ctrl"}))]
    assert parser.feed(b"\x1b[24~") == [KeyInput("f12")]
    assert parser.feed(b"\x1b[<0;12") == []
    assert parser.feed(b";7M") == [PointerInput("press", 12, 7, button="left")]
    assert parser.feed(b"\x1b[<32;13;8M") == [
        PointerInput("move", 13, 8, button="left")
    ]
    assert parser.feed(b"\x1b[<0;13;8m") == [
        PointerInput("release", 13, 8, button="left")
    ]
    assert parser.feed(b"\x1b[<64;13;8M") == [PointerInput("scroll", 13, 8, scroll_y=1)]


def test_null_sink_and_factory_are_explicit_no_ops() -> None:
    sink = create_input_sink("none")

    assert isinstance(sink, NullInputSink)
    assert not sink.send_key(KeyInput("a"))
    assert not sink.send_pointer(PointerInput("move", 1, 1), 0, 0)
    with pytest.raises(ValueError, match="auto, pynput, or none"):
        create_input_sink("telepathy")


class FakeKeyboardController:
    def __init__(self) -> None:
        self.operations: list[tuple[str, object]] = []

    def press(self, key: object) -> None:
        self.operations.append(("press", key))

    def release(self, key: object) -> None:
        self.operations.append(("release", key))


class FakeMouseController:
    def __init__(self) -> None:
        self.operations: list[tuple[object, ...]] = []
        self._position = (0, 0)

    @property
    def position(self) -> tuple[int, int]:
        return self._position

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        self._position = value
        self.operations.append(("position", *value))

    def press(self, button: object) -> None:
        self.operations.append(("press", button))

    def release(self, button: object) -> None:
        self.operations.append(("release", button))

    def scroll(self, x: int, y: int) -> None:
        self.operations.append(("scroll", x, y))


def test_pynput_sink_targets_display_and_releases_held_buttons(monkeypatch) -> None:
    keyboard_controller = FakeKeyboardController()
    mouse_controller = FakeMouseController()
    environments: list[tuple[str | None, str | None]] = []
    keyboard = SimpleNamespace(
        Key=SimpleNamespace(
            ctrl="CTRL",
            alt="ALT",
            shift="SHIFT",
            enter="ENTER",
        ),
        Controller=lambda: (
            environments.append(
                (os.environ.get("DISPLAY"), os.environ.get("PYNPUT_BACKEND"))
            )
            or keyboard_controller
        ),
    )
    mouse = SimpleNamespace(
        Button=SimpleNamespace(left="LEFT", middle="MIDDLE", right="RIGHT"),
        Controller=lambda: mouse_controller,
    )

    def fake_import(name: str):
        return keyboard if name == "pynput.keyboard" else mouse

    monkeypatch.setattr(input_module.importlib, "import_module", fake_import)
    monkeypatch.setattr(input_module, "_enable_windows_dpi_awareness", lambda: None)
    original_display = os.environ.get("DISPLAY")
    original_backend = os.environ.get("PYNPUT_BACKEND")

    sink = input_module.PynputInputSink(display_name=":42")
    assert sink.name == "pynput[:42]"
    assert environments == [(":42", "xorg")]
    assert os.environ.get("DISPLAY") == original_display
    assert os.environ.get("PYNPUT_BACKEND") == original_backend

    assert sink.send_key(KeyInput("c", frozenset({"ctrl"})))
    assert sink.send_pointer(PointerInput("press", 1, 1, button="left"), 10, 20)
    sink.close()

    assert keyboard_controller.operations == [
        ("press", "CTRL"),
        ("press", "c"),
        ("release", "c"),
        ("release", "CTRL"),
    ]
    assert mouse_controller.operations == [
        ("position", 10, 20),
        ("press", "LEFT"),
        ("release", "LEFT"),
    ]


def test_live_session_updates_control_viewport_and_closes_router(monkeypatch) -> None:
    instances = []

    class FakeTerminalInputPump:
        def __init__(self, router, **_options) -> None:
            self.router = router
            self.stop_requested = threading.Event()
            self.escape_requested = False
            instances.append(self)

        def start(self):
            return self

        def raise_if_failed(self) -> None:
            pass

        def stop(self) -> None:
            self.router.close()

    monkeypatch.setattr(session_module, "TerminalInputPump", FakeTerminalInputPump)
    sink = FakeSink()
    router = InputRouter(sink, CaptureRegion(0, 0, 8, 8))
    source = IterableFrameSource([np.zeros((8, 8, 3), dtype=np.uint8)])
    renderer = FrameRenderer(RenderConfig(width=4, height=2, charset="@"))

    stats = run_terminal_session(
        source,
        renderer,
        TerminalSessionConfig(max_frames=1, alternate_screen=False, show_stats=False),
        output=io.StringIO(),
        input_router=router,
    )

    assert instances
    assert router.viewport == RenderViewport(4, 2)
    assert stats.input_events == 0
    assert sink.closes == 1


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_terminal_input_pump_routes_keys_and_hard_stops_before_injection(
    monkeypatch,
) -> None:
    class FakeReader:
        def __init__(self) -> None:
            self.chunks = [b"a\x1d"]
            self.started = False
            self.closed = False

        def start(self) -> None:
            self.started = True

        def read(self, _timeout: float) -> bytes | None:
            if self.chunks:
                return self.chunks.pop(0)
            time.sleep(0.001)
            return None

        def close(self) -> None:
            self.closed = True

    reader = FakeReader()
    monkeypatch.setattr(
        "glyph_forge.live.terminal_input._terminal_reader", lambda _stream: reader
    )
    sink = FakeSink()
    router = InputRouter(sink, CaptureRegion(0, 0, 10, 10))
    output = TTYBuffer()
    pump = TerminalInputPump(router, input_stream=TTYBuffer(), output=output).start()

    assert pump.stop_requested.wait(1)
    pump.stop()

    assert reader.started and reader.closed
    assert sink.keys == [KeyInput("a")]
    assert pump.escape_requested
    assert output.getvalue().startswith(TerminalInputPump.ENABLE_MOUSE)
    assert output.getvalue().endswith(TerminalInputPump.DISABLE_MOUSE)

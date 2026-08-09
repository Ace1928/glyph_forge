"""Permission-aware keyboard and pointer routing for live glyph surfaces.

Capture and input injection deliberately use separate interfaces.  Importing
this module has no native side effects; the optional controller backend is
loaded only after a caller explicitly requests control.
"""

from __future__ import annotations

import importlib
import os
import platform
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from .capture import CaptureRegion

Modifier: TypeAlias = Literal["shift", "alt", "ctrl"]
PointerAction: TypeAlias = Literal["move", "press", "release", "scroll"]
PointerButton: TypeAlias = Literal["left", "middle", "right"]


class InputRoutingError(RuntimeError):
    """Raised when explicit input forwarding cannot be started or completed."""


class InputBackendUnavailable(InputRoutingError):
    """Raised when an optional input-injection backend is unavailable."""


@dataclass(frozen=True, slots=True)
class KeyInput:
    """One terminal key tap destined for the controlled display."""

    key: str
    modifiers: frozenset[Modifier] = frozenset()


@dataclass(frozen=True, slots=True)
class PointerInput:
    """One pointer event expressed in one-based terminal cell coordinates."""

    action: PointerAction
    column: int
    row: int
    button: PointerButton | None = None
    scroll_x: int = 0
    scroll_y: int = 0
    modifiers: frozenset[Modifier] = frozenset()


InputEvent: TypeAlias = KeyInput | PointerInput


@dataclass(frozen=True, slots=True)
class RenderViewport:
    """Terminal-cell rectangle occupied by a rendered frame."""

    columns: int
    rows: int
    left: int = 1
    top: int = 1

    def __post_init__(self) -> None:
        if self.columns < 1 or self.rows < 1:
            raise ValueError("Viewport dimensions must be positive")
        if self.left < 1 or self.top < 1:
            raise ValueError("Viewport origins use positive terminal coordinates")


def map_pointer_to_capture(
    column: int,
    row: int,
    viewport: RenderViewport,
    capture: CaptureRegion,
) -> tuple[int, int] | None:
    """Map a terminal cell center into absolute capture-space pixels.

    Coordinates outside the rendered rectangle are ignored.  This prevents a
    status line, terminal padding, or future letterboxing from becoming an
    accidental click target.
    """

    local_x = column - viewport.left
    local_y = row - viewport.top
    if not 0 <= local_x < viewport.columns or not 0 <= local_y < viewport.rows:
        return None
    relative_x = (local_x + 0.5) / viewport.columns
    relative_y = (local_y + 0.5) / viewport.rows
    x = capture.left + min(capture.width - 1, int(relative_x * capture.width))
    y = capture.top + min(capture.height - 1, int(relative_y * capture.height))
    return x, y


@runtime_checkable
class InputSink(Protocol):
    """Platform adapter receiving already-mapped input events."""

    @property
    def name(self) -> str:
        """Human-readable backend name."""

    def send_key(self, event: KeyInput) -> bool:
        """Inject one key tap, returning whether it was understood."""

    def send_pointer(self, event: PointerInput, x: int, y: int) -> bool:
        """Inject one mapped pointer event."""

    def release_all(self) -> None:
        """Release every button or modifier retained by the backend."""

    def close(self) -> None:
        """Release native resources. Calling this repeatedly is safe."""


class NullInputSink:
    """Side-effect-free sink used by embedders and capability probes."""

    name = "none"

    def send_key(self, _event: KeyInput) -> bool:
        return False

    def send_pointer(self, _event: PointerInput, _x: int, _y: int) -> bool:
        return False

    def release_all(self) -> None:
        pass

    def close(self) -> None:
        pass


_SPECIAL_KEYS = {
    "backspace": "backspace",
    "delete": "delete",
    "down": "down",
    "end": "end",
    "enter": "enter",
    "escape": "esc",
    "home": "home",
    "insert": "insert",
    "left": "left",
    "page_down": "page_down",
    "page_up": "page_up",
    "right": "right",
    "tab": "tab",
    "up": "up",
    **{f"f{index}": f"f{index}" for index in range(1, 13)},
}


@contextmanager
def _temporary_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _enable_windows_dpi_awareness() -> None:
    """Align controller and capture coordinates on scaled Windows desktops."""

    if platform.system() != "Windows":
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        # PER_MONITOR_AWARE_V2; failure is harmless if another UI set it first.
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        pass


class PynputInputSink:
    """Lazy cross-platform controller backed by the optional pynput package."""

    def __init__(self, *, display_name: str | None = None) -> None:
        _enable_windows_dpi_awareness()
        environment: dict[str, str] = {}
        if display_name is not None:
            environment = {
                "DISPLAY": display_name,
                "PYNPUT_BACKEND": "xorg",
            }
        try:
            with _temporary_environment(environment):
                keyboard = importlib.import_module("pynput.keyboard")
                mouse = importlib.import_module("pynput.mouse")
                self._keyboard = keyboard.Controller()
                self._mouse = mouse.Controller()
        except Exception as exc:
            target = f" for display {display_name}" if display_name else ""
            raise InputBackendUnavailable(
                "Desktop control requires pynput and OS input permission"
                f"{target}; install glyph-forge[control] and grant Accessibility "
                f"or input-control access ({exc})"
            ) from exc
        self._key_namespace = keyboard.Key
        self._button_namespace = mouse.Button
        self._pressed_buttons: set[PointerButton] = set()
        self._closed = False
        self._lock = threading.RLock()
        self._display_name = display_name

    @property
    def name(self) -> str:
        suffix = f"[{self._display_name}]" if self._display_name else ""
        return f"pynput{suffix}"

    def _key(self, name: str) -> Any | None:
        special = _SPECIAL_KEYS.get(name)
        if special is not None:
            return getattr(self._key_namespace, special, None)
        return name if len(name) == 1 else None

    def _modifier_keys(self, modifiers: frozenset[Modifier]) -> list[Any]:
        names = (("ctrl", "ctrl"), ("alt", "alt"), ("shift", "shift"))
        return [
            key
            for modifier, attribute in names
            if modifier in modifiers
            for key in [getattr(self._key_namespace, attribute, None)]
            if key is not None
        ]

    @contextmanager
    def _held_modifiers(self, modifiers: frozenset[Modifier]) -> Iterator[None]:
        pressed: list[Any] = []
        try:
            for key in self._modifier_keys(modifiers):
                self._keyboard.press(key)
                pressed.append(key)
            yield
        finally:
            for key in reversed(pressed):
                try:
                    self._keyboard.release(key)
                except Exception:
                    pass

    def send_key(self, event: KeyInput) -> bool:
        key = self._key(event.key)
        if key is None:
            return False
        with self._lock:
            if self._closed:
                return False
            try:
                with self._held_modifiers(event.modifiers):
                    self._keyboard.press(key)
                    self._keyboard.release(key)
            except Exception as exc:
                raise InputRoutingError(f"Keyboard injection failed: {exc}") from exc
        return True

    def send_pointer(self, event: PointerInput, x: int, y: int) -> bool:
        with self._lock:
            if self._closed:
                return False
            button = (
                getattr(self._button_namespace, event.button, None)
                if event.button is not None
                else None
            )
            try:
                with self._held_modifiers(event.modifiers):
                    self._mouse.position = (x, y)
                    if (
                        event.action == "press"
                        and button is not None
                        and event.button is not None
                    ):
                        self._mouse.press(button)
                        self._pressed_buttons.add(event.button)
                    elif (
                        event.action == "release"
                        and button is not None
                        and event.button is not None
                    ):
                        self._mouse.release(button)
                        self._pressed_buttons.discard(event.button)
                    elif event.action == "scroll":
                        self._mouse.scroll(event.scroll_x, event.scroll_y)
                    elif event.action != "move":
                        return False
            except Exception as exc:
                raise InputRoutingError(f"Pointer injection failed: {exc}") from exc
        return True

    def release_all(self) -> None:
        with self._lock:
            for name in tuple(self._pressed_buttons):
                button = getattr(self._button_namespace, name, None)
                if button is not None:
                    try:
                        self._mouse.release(button)
                    except Exception:
                        pass
            self._pressed_buttons.clear()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.release_all()
            self._closed = True


class InputRouter:
    """Map viewport events into one explicitly selected capture target."""

    def __init__(self, sink: InputSink, capture_region: CaptureRegion) -> None:
        self.sink = sink
        self.capture_region = capture_region
        self._viewport: RenderViewport | None = None
        self._lock = threading.RLock()
        self._closed = False
        self.routed_events = 0

    @property
    def name(self) -> str:
        return self.sink.name

    @property
    def viewport(self) -> RenderViewport | None:
        with self._lock:
            return self._viewport

    def update_viewport(self, columns: int, rows: int) -> None:
        with self._lock:
            self._viewport = RenderViewport(columns, rows)

    def route(self, event: InputEvent) -> bool:
        with self._lock:
            if self._closed:
                return False
            if isinstance(event, KeyInput):
                routed = self.sink.send_key(event)
            else:
                viewport = self._viewport
                if viewport is None:
                    return False
                target = map_pointer_to_capture(
                    event.column,
                    event.row,
                    viewport,
                    self.capture_region,
                )
                if target is None:
                    return False
                routed = self.sink.send_pointer(event, *target)
            if routed:
                self.routed_events += 1
            return routed

    def release_all(self) -> None:
        with self._lock:
            self.sink.release_all()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self.sink.release_all()
            finally:
                try:
                    self.sink.close()
                finally:
                    self._closed = True


def create_input_sink(
    backend: str = "auto",
    *,
    display_name: str | None = None,
) -> InputSink:
    """Create an input sink only after an explicit control request."""

    selected = backend.casefold()
    if selected == "none":
        return NullInputSink()
    if selected in {"auto", "pynput"}:
        return PynputInputSink(display_name=display_name)
    raise ValueError("Input backend must be auto, pynput, or none")


__all__ = [
    "InputBackendUnavailable",
    "InputEvent",
    "InputRouter",
    "InputRoutingError",
    "InputSink",
    "KeyInput",
    "Modifier",
    "NullInputSink",
    "PointerAction",
    "PointerButton",
    "PointerInput",
    "PynputInputSink",
    "RenderViewport",
    "create_input_sink",
    "map_pointer_to_capture",
]

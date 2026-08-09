"""Raw terminal input capture for explicitly controlled live sessions."""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, TextIO

from .input import (
    InputEvent,
    InputRouter,
    InputRoutingError,
    KeyInput,
    Modifier,
    PointerButton,
    PointerInput,
)

_MOUSE_PATTERN = re.compile(rb"^\x1b\[<(\d+);(\d+);(\d+)([Mm])")
_CSI_PATTERN = re.compile(rb"^\x1b\[([0-9;]*)([@-~])")
_SS3_PATTERN = re.compile(rb"^\x1bO([A-Za-z])")

_CSI_KEYS = {
    "A": "up",
    "B": "down",
    "C": "right",
    "D": "left",
    "F": "end",
    "H": "home",
    "Z": "tab",
}
_TILDE_KEYS = {
    1: "home",
    2: "insert",
    3: "delete",
    4: "end",
    5: "page_up",
    6: "page_down",
    7: "home",
    8: "end",
    11: "f1",
    12: "f2",
    13: "f3",
    14: "f4",
    15: "f5",
    17: "f6",
    18: "f7",
    19: "f8",
    20: "f9",
    21: "f10",
    23: "f11",
    24: "f12",
}
_SS3_KEYS = {
    "A": "up",
    "B": "down",
    "C": "right",
    "D": "left",
    "F": "end",
    "H": "home",
    "P": "f1",
    "Q": "f2",
    "R": "f3",
    "S": "f4",
}
_CONTROL_CHARACTERS = {
    0: " ",
    **{value: chr(ord("a") + value - 1) for value in range(1, 27)},
    28: "\\",
    30: "^",
    31: "_",
}
_WINDOWS_EXTENDED_KEYS = {
    "G": b"\x1b[H",
    "H": b"\x1b[A",
    "I": b"\x1b[5~",
    "K": b"\x1b[D",
    "M": b"\x1b[C",
    "O": b"\x1b[F",
    "P": b"\x1b[B",
    "Q": b"\x1b[6~",
    "R": b"\x1b[2~",
    "S": b"\x1b[3~",
    ";": b"\x1bOP",
    "<": b"\x1bOQ",
    "=": b"\x1bOR",
    ">": b"\x1bOS",
}


@dataclass(frozen=True, slots=True)
class EmergencyInput:
    """Hard stop emitted for Ctrl+] before any platform injection occurs."""


ParsedInput = InputEvent | EmergencyInput


def _modifiers_from_code(code: int) -> frozenset[Modifier]:
    bits = max(0, code - 1)
    modifiers: set[Modifier] = set()
    if bits & 1:
        modifiers.add("shift")
    if bits & 2:
        modifiers.add("alt")
    if bits & 4:
        modifiers.add("ctrl")
    return frozenset(modifiers)


def _mouse_modifiers(code: int) -> frozenset[Modifier]:
    modifiers: set[Modifier] = set()
    if code & 4:
        modifiers.add("shift")
    if code & 8:
        modifiers.add("alt")
    if code & 16:
        modifiers.add("ctrl")
    return frozenset(modifiers)


def _decode_mouse(match: re.Match[bytes]) -> PointerInput | None:
    code = int(match.group(1))
    column = int(match.group(2))
    row = int(match.group(3))
    final = match.group(4)
    modifiers = _mouse_modifiers(code)
    button_index = code & 3
    buttons: dict[int, PointerButton] = {0: "left", 1: "middle", 2: "right"}
    button = buttons.get(button_index)

    if code & 64:
        scroll_x = 0
        scroll_y = 0
        if button_index == 0:
            scroll_y = 1
        elif button_index == 1:
            scroll_y = -1
        elif button_index == 2:
            scroll_x = -1
        else:
            scroll_x = 1
        return PointerInput(
            "scroll",
            column,
            row,
            scroll_x=scroll_x,
            scroll_y=scroll_y,
            modifiers=modifiers,
        )
    if code & 32:
        return PointerInput(
            "move",
            column,
            row,
            button=button,
            modifiers=modifiers,
        )
    if button is None:
        return None
    return PointerInput(
        "release" if final == b"m" else "press",
        column,
        row,
        button=button,
        modifiers=modifiers,
    )


def _utf8_size(first: int) -> int:
    if first < 0x80:
        return 1
    if first & 0xE0 == 0xC0:
        return 2
    if first & 0xF0 == 0xE0:
        return 3
    if first & 0xF8 == 0xF0:
        return 4
    return 1


class TerminalEventParser:
    """Incrementally parse UTF-8 keys and xterm SGR mouse reports."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    @property
    def pending(self) -> bool:
        return bool(self._buffer)

    def feed(self, data: bytes) -> list[ParsedInput]:
        self._buffer.extend(data)
        return self._drain(allow_incomplete=False)

    def flush(self) -> list[ParsedInput]:
        """Resolve an idle standalone Escape and discard malformed fragments."""

        return self._drain(allow_incomplete=True)

    def _drain(self, *, allow_incomplete: bool) -> list[ParsedInput]:
        events: list[ParsedInput] = []
        while self._buffer:
            event, consumed, incomplete = self._parse_one()
            if incomplete:
                if not allow_incomplete:
                    break
                if self._buffer[0] == 0x1B:
                    event, consumed = KeyInput("escape"), 1
                else:
                    consumed = 1
            del self._buffer[:consumed]
            if event is not None:
                events.append(event)
        return events

    def _parse_one(self) -> tuple[ParsedInput | None, int, bool]:
        first = self._buffer[0]
        if first == 0x1D:
            return EmergencyInput(), 1, False
        if first == 0x1B:
            return self._parse_escape()
        if first in {8, 127}:
            return KeyInput("backspace"), 1, False
        if first in {10, 13}:
            return KeyInput("enter"), 1, False
        if first == 9:
            return KeyInput("tab"), 1, False
        if first in _CONTROL_CHARACTERS:
            return (
                KeyInput(
                    _CONTROL_CHARACTERS[first],
                    frozenset({"ctrl"}),
                ),
                1,
                False,
            )
        return self._parse_character(offset=0, modifiers=frozenset())

    def _parse_escape(self) -> tuple[ParsedInput | None, int, bool]:
        if len(self._buffer) == 1:
            return None, 0, True
        raw = bytes(self._buffer)
        mouse = _MOUSE_PATTERN.match(raw)
        if mouse is not None:
            return _decode_mouse(mouse), mouse.end(), False
        if raw.startswith(b"\x1b[<"):
            return None, 0, True
        csi = _CSI_PATTERN.match(raw)
        if csi is not None:
            params_text = csi.group(1).decode("ascii")
            final = csi.group(2).decode("ascii")
            params = (
                [int(item) if item else 1 for item in params_text.split(";")]
                if params_text
                else []
            )
            modifier_code = params[-1] if len(params) > 1 else 1
            modifiers = _modifiers_from_code(modifier_code)
            if final == "Z":
                modifiers = frozenset(set(modifiers) | {"shift"})
            key = _CSI_KEYS.get(final)
            if final == "~" and params:
                key = _TILDE_KEYS.get(params[0])
            event = KeyInput(key, modifiers) if key is not None else None
            return event, csi.end(), False
        if raw.startswith(b"\x1b["):
            return None, 0, True
        ss3 = _SS3_PATTERN.match(raw)
        if ss3 is not None:
            key = _SS3_KEYS.get(ss3.group(1).decode("ascii"))
            event = KeyInput(key) if key is not None else None
            return event, ss3.end(), False
        if raw.startswith(b"\x1bO"):
            return None, 0, True
        return self._parse_character(offset=1, modifiers=frozenset({"alt"}))

    def _parse_character(
        self,
        *,
        offset: int,
        modifiers: frozenset[Modifier],
    ) -> tuple[ParsedInput | None, int, bool]:
        if len(self._buffer) <= offset:
            return None, 0, True
        size = _utf8_size(self._buffer[offset])
        end = offset + size
        if len(self._buffer) < end:
            return None, 0, True
        try:
            character = bytes(self._buffer[offset:end]).decode("utf-8")
        except UnicodeDecodeError:
            return None, max(1, end), False
        return KeyInput(character, modifiers), end, False


class _TerminalReader(Protocol):
    def start(self) -> None: ...

    def read(self, timeout: float) -> bytes | None: ...

    def close(self) -> None: ...


class _PosixTerminalReader:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.fd = stream.fileno()
        self._attributes: Any | None = None

    def start(self) -> None:
        import termios
        import tty

        self._attributes = termios.tcgetattr(self.fd)
        tty.setraw(self.fd, when=termios.TCSANOW)

    def read(self, timeout: float) -> bytes | None:
        import select

        readable, _, _ = select.select([self.fd], [], [], timeout)
        if not readable:
            return None
        return os.read(self.fd, 4096)

    def close(self) -> None:
        if self._attributes is None:
            return
        import termios

        termios.tcsetattr(self.fd, termios.TCSADRAIN, self._attributes)
        self._attributes = None


class _WindowsTerminalReader:
    def __init__(self, _stream: TextIO) -> None:
        self._handle: int | None = None
        self._mode: int | None = None

    def start(self) -> None:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = int(kernel32.GetStdHandle(-10))
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            raise OSError("stdin is not a Windows console input handle")
        # Raw character input plus VT key sequences. Mouse-capable terminal
        # emulators return SGR reports after Glyph Forge enables mode 1006.
        new_mode = (mode.value | 0x0200 | 0x0080) & ~(0x0001 | 0x0002 | 0x0004)
        if not kernel32.SetConsoleMode(handle, new_mode):
            raise OSError("Windows could not enable virtual-terminal input")
        self._handle = handle
        self._mode = mode.value

    def read(self, timeout: float) -> bytes | None:
        msvcrt = import_module("msvcrt")

        deadline = time.monotonic() + timeout
        characters: list[str] = []
        while time.monotonic() < deadline:
            if not msvcrt.kbhit():
                if characters:
                    break
                time.sleep(min(0.005, max(0.0, deadline - time.monotonic())))
                continue
            character = msvcrt.getwch()
            if character in {"\x00", "\xe0"}:
                extended = msvcrt.getwch()
                mapped = _WINDOWS_EXTENDED_KEYS.get(extended)
                if mapped is not None:
                    characters.append(mapped.decode("ascii"))
            else:
                characters.append(character)
        if not characters:
            return None
        return "".join(characters).encode("utf-8", errors="surrogatepass")

    def close(self) -> None:
        if self._handle is None or self._mode is None:
            return
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetConsoleMode(self._handle, self._mode)
        self._handle = None
        self._mode = None


def _terminal_reader(stream: TextIO) -> _TerminalReader:
    if os.name == "nt":
        return _WindowsTerminalReader(stream)
    return _PosixTerminalReader(stream)


class TerminalInputPump:
    """Route terminal input on a daemon thread without delaying rendering."""

    ENABLE_MOUSE = "\x1b[?1003h\x1b[?1006h"
    DISABLE_MOUSE = "\x1b[?1006l\x1b[?1003l"

    def __init__(
        self,
        router: InputRouter,
        *,
        input_stream: TextIO | None = None,
        output: TextIO | None = None,
    ) -> None:
        self.router = router
        self.input_stream = input_stream or sys.stdin
        self.output = output or sys.stdout
        self.stop_requested = threading.Event()
        self.escape_requested = False
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._reader: _TerminalReader | None = None
        self._error: BaseException | None = None

    @property
    def routed_events(self) -> int:
        return self.router.routed_events

    def start(self) -> "TerminalInputPump":
        if self._thread is not None:
            return self
        if not self.input_stream.isatty() or not self.output.isatty():
            raise InputRoutingError(
                "Interactive control requires terminal stdin and stdout"
            )
        reader = _terminal_reader(self.input_stream)
        try:
            reader.start()
            self.output.write(self.ENABLE_MOUSE)
            self.output.flush()
        except BaseException:
            try:
                reader.close()
            except Exception:
                pass
            raise
        self._reader = reader
        self._thread = threading.Thread(
            target=self._run,
            name="glyph-forge-terminal-input",
            daemon=True,
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        assert self._reader is not None
        parser = TerminalEventParser()
        try:
            while not self._shutdown.is_set():
                data = self._reader.read(0.03)
                if data == b"":
                    self.stop_requested.set()
                    break
                if data is None:
                    events = parser.flush() if parser.pending else []
                else:
                    events = parser.feed(data)
                for event in events:
                    if isinstance(event, EmergencyInput):
                        self.router.release_all()
                        self.escape_requested = True
                        self.stop_requested.set()
                        return
                    self.router.route(event)
        except BaseException as exc:
            if not self._shutdown.is_set():
                self._error = exc
                self.stop_requested.set()

    def raise_if_failed(self) -> None:
        if self._error is None:
            return
        if isinstance(self._error, Exception):
            raise self._error
        raise RuntimeError("Terminal input routing stopped unexpectedly")

    def stop(self) -> None:
        self._shutdown.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.2)
        try:
            self.output.write(self.DISABLE_MOUSE)
            self.output.flush()
        finally:
            try:
                if self._reader is not None:
                    self._reader.close()
            finally:
                try:
                    self.router.close()
                finally:
                    self._reader = None
                    self._thread = None


__all__ = [
    "EmergencyInput",
    "ParsedInput",
    "TerminalEventParser",
    "TerminalInputPump",
]

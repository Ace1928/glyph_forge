"""Low-latency live sessions and terminal presentation."""

from __future__ import annotations

import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TextIO

from .capture import FrameSource, LatestFramePump
from .input import InputRouter
from .renderers import FrameRenderer, RenderResult
from .terminal_input import TerminalInputPump


class TerminalRedraw(str, Enum):
    """Strategy used to update a live terminal surface."""

    AUTO = "auto"
    FULL = "full"
    DELTA = "delta"


def _normalize_redraw(value: TerminalRedraw | str) -> TerminalRedraw:
    if isinstance(value, TerminalRedraw):
        return value
    if not isinstance(value, str):
        raise ValueError("redraw must be auto, full, or delta")
    try:
        return TerminalRedraw(value.casefold())
    except ValueError as exc:
        raise ValueError("redraw must be auto, full, or delta") from exc


@dataclass(frozen=True, slots=True)
class TerminalSessionConfig:
    """Presentation controls for one terminal live session."""

    target_fps: float = 20.0
    duration: float | None = None
    max_frames: int | None = None
    alternate_screen: bool = True
    show_stats: bool = True
    redraw: TerminalRedraw | str = TerminalRedraw.AUTO
    fit_terminal: bool = True

    def validated(self) -> "TerminalSessionConfig":
        if self.target_fps <= 0:
            raise ValueError("target_fps must be greater than zero")
        if self.duration is not None and self.duration <= 0:
            raise ValueError("duration must be greater than zero")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be positive")
        _normalize_redraw(self.redraw)
        return self


@dataclass(frozen=True, slots=True)
class LiveSessionStats:
    """Measured results from a completed live presentation."""

    source: str
    captured_frames: int
    presented_frames: int
    dropped_frames: int
    elapsed: float
    interrupted: bool = False
    input_events: int = 0
    control_escape: bool = False
    output_bytes: int = 0
    full_redraws: int = 0
    delta_redraws: int = 0
    skipped_redraws: int = 0

    @property
    def presentation_fps(self) -> float:
        return self.presented_frames / self.elapsed if self.elapsed > 0 else 0.0


class TerminalPresenter:
    """Stateful, bandwidth-adaptive ANSI frame presenter.

    Delta updates are only used on an alternate-screen terminal. Redirected
    output deliberately remains a sequence of complete frames, preserving its
    useful line-oriented behavior.
    """

    def __init__(
        self,
        stream: TextIO,
        *,
        alternate_screen: bool,
        redraw: TerminalRedraw | str = TerminalRedraw.AUTO,
    ) -> None:
        self.stream = stream
        self.alternate_screen = alternate_screen
        self.redraw = _normalize_redraw(redraw)
        self.output_bytes = 0
        self.full_redraws = 0
        self.delta_redraws = 0
        self.skipped_redraws = 0
        self._opened = False
        self._closed = False
        self._dimensions: tuple[int, int] | None = None
        self._rows: tuple[str, ...] | None = None
        self._footer: str | None = None

    @staticmethod
    def _byte_length(text: str) -> int:
        return len(text.encode("utf-8"))

    def _write(self, payload: str) -> None:
        if not payload:
            return
        self.stream.write(payload)
        self.stream.flush()
        self.output_bytes += self._byte_length(payload)

    def open(self) -> None:
        if self._opened:
            return
        self._opened = True
        if self.alternate_screen:
            self._write("\x1b[?1049h\x1b[?25l\x1b[2J")

    def _full_payload(self, result: RenderResult, footer: str | None) -> str:
        if not self.alternate_screen:
            return result.text + (f"\n{footer}" if footer is not None else "") + "\n"
        return (
            "\x1b[H"
            + result.text
            + (f"\n{footer}" if footer is not None else "")
            + "\x1b[J"
        )

    def _delta_payload(
        self,
        rows: tuple[str, ...],
        footer: str | None,
    ) -> str:
        assert self._rows is not None
        parts: list[str] = []
        for row_number, (previous, current) in enumerate(
            zip(self._rows, rows, strict=True),
            start=1,
        ):
            if current != previous:
                parts.append(f"\x1b[{row_number};1H{current}\x1b[K")

        if footer != self._footer:
            footer_row = len(rows) + 1
            if footer is None:
                parts.append(f"\x1b[{footer_row};1H\x1b[2K")
            else:
                parts.append(f"\x1b[{footer_row};1H{footer}\x1b[K")
        return "".join(parts)

    def present(self, result: RenderResult, footer: str | None = None) -> None:
        """Write one frame using the cheapest allowed redraw representation."""

        if not self._opened:
            self.open()
        rows = tuple(result.text.split("\n"))
        dimensions = (result.width, result.height)
        full_payload = self._full_payload(result, footer)
        can_delta = (
            self.alternate_screen
            and self.redraw is not TerminalRedraw.FULL
            and self._rows is not None
            and self._dimensions == dimensions
            and len(self._rows) == len(rows)
        )

        if can_delta:
            delta_payload = self._delta_payload(rows, footer)
            use_delta = self.redraw is TerminalRedraw.DELTA or (
                self._byte_length(delta_payload) < self._byte_length(full_payload)
            )
            if use_delta:
                if delta_payload:
                    self._write(delta_payload)
                    self.delta_redraws += 1
                else:
                    self.skipped_redraws += 1
            else:
                self._write(full_payload)
                self.full_redraws += 1
        else:
            self._write(full_payload)
            self.full_redraws += 1

        self._dimensions = dimensions
        self._rows = rows
        self._footer = footer

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.alternate_screen and self._opened:
            self._write("\x1b[0m\x1b[?25h\x1b[?1049l")


def run_terminal_session(
    source: FrameSource,
    renderer: FrameRenderer,
    config: TerminalSessionConfig | None = None,
    *,
    output: TextIO | None = None,
    stop_when: Callable[[], bool] | None = None,
    input_router: InputRouter | None = None,
    input_stream: TextIO | None = None,
) -> LiveSessionStats:
    """Render the latest available source frame in an ANSI terminal surface."""

    selected = (config or TerminalSessionConfig()).validated()
    stream = output or sys.stdout
    use_alternate_screen = selected.alternate_screen and bool(
        getattr(stream, "isatty", lambda: False)()
    )
    presenter = TerminalPresenter(
        stream,
        alternate_screen=use_alternate_screen,
        redraw=selected.redraw,
    )
    interval = 1 / selected.target_fps
    started = time.monotonic()
    next_presentation = started
    deadline = started + selected.duration if selected.duration is not None else None
    sequence = 0
    presented = 0
    dropped = 0
    interrupted = False
    control_escape = False
    input_pump: TerminalInputPump | None = None

    pump = LatestFramePump(source)
    try:
        presenter.open()
        pump.start()
        if input_router is not None:
            input_pump = TerminalInputPump(
                input_router,
                input_stream=input_stream,
                output=stream,
            ).start()
        while True:
            if input_pump is not None and input_pump.stop_requested.is_set():
                input_pump.raise_if_failed()
                control_escape = input_pump.escape_requested
                break
            if stop_when is not None and stop_when():
                break
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break
            if now < next_presentation:
                pause = next_presentation - now
                if deadline is not None:
                    pause = min(pause, max(0.0, deadline - now))
                time.sleep(pause)
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    break
            timeout = interval
            if deadline is not None:
                timeout = max(0.0, min(timeout, deadline - now))
            frame = pump.next_frame(sequence, timeout=timeout)
            if frame is None:
                if pump.ended:
                    break
                continue

            dropped += max(0, frame.sequence - sequence - 1)
            sequence = frame.sequence
            max_width = None
            max_height = None
            if use_alternate_screen and selected.fit_terminal:
                terminal = shutil.get_terminal_size(
                    (renderer.config.width + 1, 30),
                )
                # Reserving the final column avoids automatic line wrapping in
                # terminals which enter wrap-pending state at the right margin.
                max_width = max(1, terminal.columns - 1)
                max_height = max(1, terminal.lines - int(selected.show_stats))
            result = renderer.render(
                frame.pixels,
                max_width=max_width,
                max_height=max_height,
            )
            if input_router is not None:
                input_router.update_viewport(result.width, result.height)
            elapsed = time.monotonic() - started
            footer = None
            if selected.show_stats:
                stats_text = (
                    f"{source.name} · {result.mode.value} · "
                    f"{presented + 1} frames · {elapsed:.1f}s · "
                    f"{dropped} dropped"
                    + (
                        f" · {input_router.routed_events} inputs · Ctrl+] exits"
                        if input_router is not None
                        else ""
                    )
                )
                if max_width is not None and len(stats_text) > max_width:
                    stats_text = (
                        stats_text[: max(1, max_width - 1)] + "…"
                        if max_width > 1
                        else stats_text[:1]
                    )
                footer = f"\x1b[2m{stats_text}\x1b[0m"
            presenter.present(result, footer)
            presented += 1
            next_presentation = max(
                next_presentation + interval,
                time.monotonic(),
            )
            if selected.max_frames is not None and presented >= selected.max_frames:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        try:
            if input_pump is not None:
                input_pump.stop()
            elif input_router is not None:
                input_router.close()
        finally:
            try:
                pump.stop()
            finally:
                presenter.close()

    return LiveSessionStats(
        source=source.name,
        captured_frames=pump.captured_frames,
        presented_frames=presented,
        dropped_frames=dropped,
        elapsed=time.monotonic() - started,
        interrupted=interrupted,
        input_events=input_router.routed_events if input_router is not None else 0,
        control_escape=control_escape,
        output_bytes=presenter.output_bytes,
        full_redraws=presenter.full_redraws,
        delta_redraws=presenter.delta_redraws,
        skipped_redraws=presenter.skipped_redraws,
    )


__all__ = [
    "LiveSessionStats",
    "TerminalPresenter",
    "TerminalRedraw",
    "TerminalSessionConfig",
    "run_terminal_session",
]

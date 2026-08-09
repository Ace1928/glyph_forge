"""Low-latency live sessions and terminal presentation."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TextIO

from .capture import FrameSource, LatestFramePump
from .input import InputRouter
from .renderers import FrameRenderer
from .terminal_input import TerminalInputPump


@dataclass(frozen=True, slots=True)
class TerminalSessionConfig:
    """Presentation controls for one terminal live session."""

    target_fps: float = 20.0
    duration: float | None = None
    max_frames: int | None = None
    alternate_screen: bool = True
    show_stats: bool = True

    def validated(self) -> "TerminalSessionConfig":
        if self.target_fps <= 0:
            raise ValueError("target_fps must be greater than zero")
        if self.duration is not None and self.duration <= 0:
            raise ValueError("duration must be greater than zero")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be positive")
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

    @property
    def presentation_fps(self) -> float:
        return self.presented_frames / self.elapsed if self.elapsed > 0 else 0.0


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

    if use_alternate_screen:
        stream.write("\x1b[?1049h\x1b[?25l\x1b[2J")
        stream.flush()

    pump = LatestFramePump(source).start()
    try:
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
            result = renderer.render(frame.pixels)
            if input_router is not None:
                input_router.update_viewport(result.width, result.height)
            elapsed = time.monotonic() - started
            prefix = "\x1b[H" if use_alternate_screen else ""
            suffix = "\x1b[J" if use_alternate_screen else "\n"
            if selected.show_stats:
                suffix = (
                    f"\n\x1b[2m{source.name} · {result.mode.value} · "
                    f"{presented + 1} frames · {elapsed:.1f}s · "
                    f"{dropped} dropped"
                    + (
                        f" · {input_router.routed_events} inputs · Ctrl+] exits"
                        if input_router is not None
                        else ""
                    )
                    + "\x1b[0m"
                    + suffix
                )
            stream.write(prefix + result.text + suffix)
            stream.flush()
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
                if use_alternate_screen:
                    stream.write("\x1b[0m\x1b[?25h\x1b[?1049l")
                    stream.flush()

    return LiveSessionStats(
        source=source.name,
        captured_frames=pump.captured_frames,
        presented_frames=presented,
        dropped_frames=dropped,
        elapsed=time.monotonic() - started,
        interrupted=interrupted,
        input_events=input_router.routed_events if input_router is not None else 0,
        control_escape=control_escape,
    )


__all__ = ["LiveSessionStats", "TerminalSessionConfig", "run_terminal_session"]

"""Live video, webcam, and desktop commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import typer
from rich.console import Console

from ..runtime import RuntimeProfile, detect_runtime_profile

console = Console(stderr=True)

app = typer.Typer(
    name="live",
    help="View cameras, videos, and desktops through low-latency glyph rendering.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _capture_dimensions(profile: RuntimeProfile) -> tuple[int, int]:
    if profile.tier.value == "eco":
        return 640, 480
    if profile.tier.value == "workstation":
        return 1920, 1080
    return 1280, 720


def _run_source(
    specification: str,
    *,
    mode: str,
    color: str,
    width: int | None,
    height: int | None,
    charset: str,
    invert: bool,
    dither: bool,
    edge_algorithm: str,
    edge_threshold: int,
    fps: float | None,
    duration: float | None,
    max_frames: int | None,
    performance: str,
    redraw: str = "auto",
    fit_terminal: bool = True,
    loop: bool = False,
    screen_backend: str = "auto",
    stop_when: Callable[[], bool] | None = None,
    control: bool = False,
    input_backend: str = "auto",
    control_display: str | None = None,
    capture_display: str | None = None,
) -> None:
    from ..live.capture import CaptureError, CaptureRegion, create_frame_source
    from ..live.input import InputRouter, InputRoutingError, create_input_sink
    from ..live.renderers import FrameRenderer, RenderConfig
    from ..live.session import TerminalSessionConfig, run_terminal_session
    from ..plugins import PluginError

    source = None
    input_router = None
    try:
        if control and control_display is None:
            raise InputRoutingError(
                "Host-desktop control from its own terminal can feed injected "
                "events back into that terminal. Use 'glyph-forge live launch "
                "--control -- COMMAND' for a safely isolated desktop or app"
            )
        profile = detect_runtime_profile(performance)
        render_width = width or profile.stream_width
        selected_color = color.casefold()
        if selected_color == "auto":
            selected_color = "truecolor" if mode == "half-block" else "ansi256"
        renderer = FrameRenderer(
            RenderConfig(
                width=render_width,
                height=height,
                mode=mode,
                color=selected_color,
                charset=charset,
                invert=invert,
                dither=dither,
                edge_algorithm=edge_algorithm,
                edge_threshold=edge_threshold,
                resample=profile.resample,
            )
        )
        target_fps = fps or profile.target_fps
        capture_width, capture_height = _capture_dimensions(profile)
        source = create_frame_source(
            specification,
            width=capture_width,
            height=capture_height,
            fps=target_fps,
            loop=loop,
            screen_backend=screen_backend,
            screen_display=capture_display,
        )
        if control:
            if not sys.stdin.isatty() or not sys.stdout.isatty():
                raise InputRoutingError(
                    "Desktop control requires focused terminal stdin and stdout"
                )
            region = getattr(source, "capture_region", None)
            if not isinstance(region, CaptureRegion):
                raise InputRoutingError(
                    "This capture backend cannot map pointer coordinates safely; "
                    "install glyph-forge[media,control] and use --backend mss"
                )
            input_router = InputRouter(
                create_input_sink(input_backend, display_name=control_display),
                region,
            )
        session = TerminalSessionConfig(
            target_fps=target_fps,
            duration=duration,
            max_frames=max_frames,
            alternate_screen=True,
            show_stats=True,
            redraw=redraw,
            fit_terminal=fit_terminal,
        ).validated()
    except (CaptureError, InputRoutingError, PluginError, ValueError) as exc:
        if source is not None:
            source.close()
        console.print(f"[bold red]Cannot start live view:[/bold red] {exc}")
        raise typer.Exit(2) from exc

    if not sys.stdout.isatty() and duration is None and max_frames is None:
        source.close()
        console.print(
            "[bold red]Live output needs a terminal.[/bold red] "
            "Use --duration or --frames for redirected output."
        )
        raise typer.Exit(2)

    console.print(
        f"[cyan]{source.name}[/cyan] · {renderer.mode.value} · "
        f"up to {renderer.config.width} columns · {target_fps:g} FPS · "
        + ("CONTROL ACTIVE · Ctrl+] to stop" if control else "Ctrl+C to stop")
    )
    try:
        stats = run_terminal_session(
            source,
            renderer,
            session,
            stop_when=stop_when,
            input_router=input_router,
        )
    except (CaptureError, InputRoutingError, OSError, PluginError) as exc:
        console.print(f"[bold red]Live session failed:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    console.print(
        f"Stopped after {stats.elapsed:.1f}s · {stats.presented_frames} displayed · "
        f"{stats.dropped_frames} stale frames dropped · "
        f"{stats.output_bytes / 1024:.1f} KiB written · "
        f"{stats.full_redraws} full/{stats.delta_redraws} delta redraws"
        + (f" · {stats.input_events} inputs routed" if control else "")
    )


@app.command("source")
def source_command(
    specification: str = typer.Argument(
        ...,
        help=(
            "Video path, URL, camera:N, screen:N, or plugin:plugin-id/source:resource"
        ),
    ),
    mode: str = typer.Option(
        "glyph",
        "--mode",
        "-m",
        help="Built-in mode or plugin:plugin-id/renderer",
    ),
    color: str = typer.Option(
        "auto", "--color", "-c", help="auto, none, ansi256, truecolor"
    ),
    width: Optional[int] = typer.Option(None, "--width", "-w", min=1),
    height: Optional[int] = typer.Option(None, "--height", min=1),
    charset: str = typer.Option("detailed", "--charset"),
    invert: bool = typer.Option(False, "--invert"),
    dither: bool = typer.Option(False, "--dither/--no-dither"),
    edge_algorithm: str = typer.Option("sobel", "--edge-algorithm"),
    edge_threshold: int = typer.Option(48, "--edge-threshold", min=0, max=255),
    fps: Optional[float] = typer.Option(None, "--fps", min=1, max=120),
    duration: Optional[float] = typer.Option(None, "--duration", min=0.01),
    frames: Optional[int] = typer.Option(None, "--frames", min=1),
    loop: bool = typer.Option(False, "--loop/--no-loop"),
    screen_backend: str = typer.Option(
        "auto", "--screen-backend", help="Screen backend: auto, mss, or pillow"
    ),
    redraw: str = typer.Option(
        "auto", "--redraw", help="Terminal updates: auto, delta, or full"
    ),
    fit_terminal: bool = typer.Option(
        True,
        "--fit/--no-fit",
        help="Fit output inside the live terminal while preserving aspect ratio",
    ),
    performance: str = typer.Option("auto", "--performance"),
) -> None:
    """Open any built-in or plugin source through one streaming command."""

    _run_source(
        specification,
        mode=mode,
        color=color,
        width=width,
        height=height,
        charset=charset,
        invert=invert,
        dither=dither,
        edge_algorithm=edge_algorithm,
        edge_threshold=edge_threshold,
        fps=fps,
        duration=duration,
        max_frames=frames,
        performance=performance,
        redraw=redraw,
        fit_terminal=fit_terminal,
        loop=loop,
        screen_backend=screen_backend,
    )


@app.command("camera")
def camera_command(
    index: int = typer.Argument(0, min=0, help="Webcam/device index."),
    mode: str = typer.Option(
        "braille",
        "--mode",
        "-m",
        help="Built-in mode or plugin:plugin-id/renderer",
    ),
    color: str = typer.Option(
        "auto", "--color", "-c", help="auto, none, ansi256, truecolor"
    ),
    width: Optional[int] = typer.Option(None, "--width", "-w", min=1),
    height: Optional[int] = typer.Option(None, "--height", min=1),
    charset: str = typer.Option("general", "--charset"),
    invert: bool = typer.Option(False, "--invert"),
    dither: bool = typer.Option(False, "--dither/--no-dither"),
    edge_algorithm: str = typer.Option("sobel", "--edge-algorithm"),
    edge_threshold: int = typer.Option(48, "--edge-threshold", min=0, max=255),
    fps: Optional[float] = typer.Option(None, "--fps", min=1, max=120),
    duration: Optional[float] = typer.Option(None, "--duration", min=0.01),
    frames: Optional[int] = typer.Option(None, "--frames", min=1),
    redraw: str = typer.Option(
        "auto", "--redraw", help="Terminal updates: auto, delta, or full"
    ),
    fit_terminal: bool = typer.Option(
        True,
        "--fit/--no-fit",
        help="Fit output inside the live terminal while preserving aspect ratio",
    ),
    performance: str = typer.Option("auto", "--performance"),
) -> None:
    """Render a webcam as responsive terminal glyph art."""

    _run_source(
        f"camera:{index}",
        mode=mode,
        color=color,
        width=width,
        height=height,
        charset=charset,
        invert=invert,
        dither=dither,
        edge_algorithm=edge_algorithm,
        edge_threshold=edge_threshold,
        fps=fps,
        duration=duration,
        max_frames=frames,
        performance=performance,
        redraw=redraw,
        fit_terminal=fit_terminal,
    )


@app.command("screen")
def screen_command(
    monitor: int = typer.Argument(
        1, min=0, help="Monitor index (0 is the combined desktop in MSS)."
    ),
    mode: str = typer.Option(
        "half-block",
        "--mode",
        "-m",
        help="Built-in mode or plugin:plugin-id/renderer",
    ),
    color: str = typer.Option(
        "auto", "--color", "-c", help="auto, none, ansi256, truecolor"
    ),
    width: Optional[int] = typer.Option(None, "--width", "-w", min=1),
    height: Optional[int] = typer.Option(None, "--height", min=1),
    charset: str = typer.Option("detailed", "--charset"),
    invert: bool = typer.Option(False, "--invert"),
    dither: bool = typer.Option(False, "--dither/--no-dither"),
    edge_algorithm: str = typer.Option("sobel", "--edge-algorithm"),
    edge_threshold: int = typer.Option(48, "--edge-threshold", min=0, max=255),
    fps: Optional[float] = typer.Option(None, "--fps", min=1, max=120),
    duration: Optional[float] = typer.Option(None, "--duration", min=0.01),
    frames: Optional[int] = typer.Option(None, "--frames", min=1),
    backend: str = typer.Option(
        "auto", "--backend", help="Screen backend: auto, mss, or pillow"
    ),
    control: bool = typer.Option(
        False,
        "--control/--view-only",
        help=(
            "Request input forwarding (safe isolated targets only; host views "
            "explain the supported alternative)."
        ),
    ),
    input_backend: str = typer.Option(
        "auto", "--input-backend", help="Input backend: auto, pynput, or none"
    ),
    redraw: str = typer.Option(
        "auto", "--redraw", help="Terminal updates: auto, delta, or full"
    ),
    fit_terminal: bool = typer.Option(
        True,
        "--fit/--no-fit",
        help="Fit output inside the live terminal while preserving aspect ratio",
    ),
    performance: str = typer.Option("auto", "--performance"),
) -> None:
    """Mirror a desktop through high-fidelity terminal glyph rendering."""

    _run_source(
        f"screen:{monitor}",
        mode=mode,
        color=color,
        width=width,
        height=height,
        charset=charset,
        invert=invert,
        dither=dither,
        edge_algorithm=edge_algorithm,
        edge_threshold=edge_threshold,
        fps=fps,
        duration=duration,
        max_frames=frames,
        performance=performance,
        redraw=redraw,
        fit_terminal=fit_terminal,
        screen_backend=backend,
        control=control,
        input_backend=input_backend,
    )


@app.command("video")
def video_command(
    source: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    mode: str = typer.Option(
        "glyph",
        "--mode",
        "-m",
        help="Built-in mode or plugin:plugin-id/renderer",
    ),
    color: str = typer.Option(
        "auto", "--color", "-c", help="auto, none, ansi256, truecolor"
    ),
    width: Optional[int] = typer.Option(None, "--width", "-w", min=1),
    height: Optional[int] = typer.Option(None, "--height", min=1),
    charset: str = typer.Option("detailed", "--charset"),
    invert: bool = typer.Option(False, "--invert"),
    dither: bool = typer.Option(False, "--dither/--no-dither"),
    edge_algorithm: str = typer.Option("sobel", "--edge-algorithm"),
    edge_threshold: int = typer.Option(48, "--edge-threshold", min=0, max=255),
    fps: Optional[float] = typer.Option(None, "--fps", min=1, max=120),
    duration: Optional[float] = typer.Option(None, "--duration", min=0.01),
    frames: Optional[int] = typer.Option(None, "--frames", min=1),
    loop: bool = typer.Option(False, "--loop/--no-loop"),
    redraw: str = typer.Option(
        "auto", "--redraw", help="Terminal updates: auto, delta, or full"
    ),
    fit_terminal: bool = typer.Option(
        True,
        "--fit/--no-fit",
        help="Fit output inside the live terminal while preserving aspect ratio",
    ),
    performance: str = typer.Option("auto", "--performance"),
) -> None:
    """Play a video directly in the terminal without preloading its frames."""

    _run_source(
        str(source),
        mode=mode,
        color=color,
        width=width,
        height=height,
        charset=charset,
        invert=invert,
        dither=dither,
        edge_algorithm=edge_algorithm,
        edge_threshold=edge_threshold,
        fps=fps,
        duration=duration,
        max_frames=frames,
        performance=performance,
        redraw=redraw,
        fit_terminal=fit_terminal,
        loop=loop,
    )


@app.command("url")
def url_command(
    source: str = typer.Argument(..., help="Video page URL supported by yt-dlp."),
    mode: str = typer.Option(
        "glyph",
        "--mode",
        "-m",
        help="Built-in mode or plugin:plugin-id/renderer",
    ),
    color: str = typer.Option(
        "auto", "--color", "-c", help="auto, none, ansi256, truecolor"
    ),
    width: Optional[int] = typer.Option(None, "--width", "-w", min=1),
    height: Optional[int] = typer.Option(None, "--height", min=1),
    charset: str = typer.Option("detailed", "--charset"),
    invert: bool = typer.Option(False, "--invert"),
    dither: bool = typer.Option(False, "--dither/--no-dither"),
    edge_algorithm: str = typer.Option("sobel", "--edge-algorithm"),
    edge_threshold: int = typer.Option(48, "--edge-threshold", min=0, max=255),
    fps: Optional[float] = typer.Option(None, "--fps", min=1, max=120),
    duration: Optional[float] = typer.Option(None, "--duration", min=0.01),
    frames: Optional[int] = typer.Option(None, "--frames", min=1),
    redraw: str = typer.Option(
        "auto", "--redraw", help="Terminal updates: auto, delta, or full"
    ),
    fit_terminal: bool = typer.Option(
        True,
        "--fit/--no-fit",
        help="Fit output inside the live terminal while preserving aspect ratio",
    ),
    performance: str = typer.Option("auto", "--performance"),
) -> None:
    """Play a supported video-site URL without downloading it first."""

    _run_source(
        f"url:{source}",
        mode=mode,
        color=color,
        width=width,
        height=height,
        charset=charset,
        invert=invert,
        dither=dither,
        edge_algorithm=edge_algorithm,
        edge_threshold=edge_threshold,
        fps=fps,
        duration=duration,
        max_frames=frames,
        performance=performance,
        redraw=redraw,
        fit_terminal=fit_terminal,
    )


@app.command(
    "launch",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def launch_command(
    command: list[str] = typer.Argument(
        ...,
        help="Application command and arguments; place -- before app options.",
    ),
    display_width: int = typer.Option(1280, "--display-width", min=64),
    display_height: int = typer.Option(720, "--display-height", min=64),
    columns: Optional[int] = typer.Option(None, "--columns", min=1),
    mode: str = typer.Option("edge", "--mode", "-m"),
    color: str = typer.Option("auto", "--color", "-c"),
    fps: Optional[float] = typer.Option(None, "--fps", min=1, max=120),
    duration: Optional[float] = typer.Option(None, "--duration", min=0.01),
    control: bool = typer.Option(
        False,
        "--control/--view-only",
        help="Forward terminal keyboard and pointer input; Ctrl+] is the hard stop.",
    ),
    input_backend: str = typer.Option(
        "auto", "--input-backend", help="Input backend: auto, pynput, or none"
    ),
    redraw: str = typer.Option(
        "auto", "--redraw", help="Terminal updates: auto, delta, or full"
    ),
    fit_terminal: bool = typer.Option(
        True,
        "--fit/--no-fit",
        help="Fit output inside the live terminal while preserving aspect ratio",
    ),
    performance: str = typer.Option("auto", "--performance"),
) -> None:
    """Launch an app in isolated Xvfb and render its display in the terminal."""

    from ..live.virtual import VirtualDisplayError, VirtualDisplaySession

    process = None
    try:
        with VirtualDisplaySession(display_width, display_height) as display:
            process = display.launch(command)
            console.print(
                f"[cyan]{display.name}[/cyan] · app PID {process.pid} · "
                "the app closes with this viewer"
            )
            _run_source(
                "screen:0",
                mode=mode,
                color=color,
                width=columns,
                height=None,
                charset="detailed",
                invert=False,
                dither=False,
                edge_algorithm="scharr",
                edge_threshold=48,
                fps=fps,
                duration=duration,
                max_frames=None,
                performance=performance,
                redraw=redraw,
                fit_terminal=fit_terminal,
                screen_backend="mss",
                stop_when=lambda: process.poll() is not None,
                control=control,
                input_backend=input_backend,
                control_display=display.name,
                capture_display=display.name,
            )
    except (VirtualDisplayError, ValueError) as exc:
        console.print(f"[bold red]Virtual display failed:[/bold red] {exc}")
        raise typer.Exit(2) from exc
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


__all__ = [
    "app",
    "camera_command",
    "launch_command",
    "screen_command",
    "source_command",
    "url_command",
    "video_command",
]

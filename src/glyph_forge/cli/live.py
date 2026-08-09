"""Live video, webcam, and desktop commands."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

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
    fps: float | None,
    duration: float | None,
    max_frames: int | None,
    performance: str,
    loop: bool = False,
    screen_backend: str = "auto",
) -> None:
    from ..live.capture import CaptureError, create_frame_source
    from ..live.renderers import FrameRenderer, RenderConfig
    from ..live.session import TerminalSessionConfig, run_terminal_session

    try:
        profile = detect_runtime_profile(performance)
        terminal = shutil.get_terminal_size((profile.stream_width, 30))
        render_width = width or min(profile.stream_width, max(20, terminal.columns))
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
        )
        session = TerminalSessionConfig(
            target_fps=target_fps,
            duration=duration,
            max_frames=max_frames,
            alternate_screen=True,
            show_stats=True,
        )
    except (CaptureError, ValueError) as exc:
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
        f"{renderer.config.width} columns · {target_fps:g} FPS · Ctrl+C to stop"
    )
    try:
        stats = run_terminal_session(source, renderer, session)
    except CaptureError as exc:
        console.print(f"[bold red]Live capture failed:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    console.print(
        f"Stopped after {stats.elapsed:.1f}s · {stats.presented_frames} displayed · "
        f"{stats.dropped_frames} stale frames dropped"
    )


@app.command("camera")
def camera_command(
    index: int = typer.Argument(0, min=0, help="Webcam/device index."),
    mode: str = typer.Option(
        "braille", "--mode", "-m", help="glyph, braille, half-block, quadrant"
    ),
    color: str = typer.Option(
        "auto", "--color", "-c", help="auto, none, ansi256, truecolor"
    ),
    width: Optional[int] = typer.Option(None, "--width", "-w", min=1),
    height: Optional[int] = typer.Option(None, "--height", min=1),
    charset: str = typer.Option("general", "--charset"),
    invert: bool = typer.Option(False, "--invert"),
    dither: bool = typer.Option(False, "--dither/--no-dither"),
    fps: Optional[float] = typer.Option(None, "--fps", min=1, max=120),
    duration: Optional[float] = typer.Option(None, "--duration", min=0.01),
    frames: Optional[int] = typer.Option(None, "--frames", min=1),
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
        fps=fps,
        duration=duration,
        max_frames=frames,
        performance=performance,
    )


@app.command("screen")
def screen_command(
    monitor: int = typer.Argument(
        1, min=0, help="Monitor index (0 is the combined desktop in MSS)."
    ),
    mode: str = typer.Option(
        "half-block", "--mode", "-m", help="glyph, braille, half-block, quadrant"
    ),
    color: str = typer.Option(
        "auto", "--color", "-c", help="auto, none, ansi256, truecolor"
    ),
    width: Optional[int] = typer.Option(None, "--width", "-w", min=1),
    height: Optional[int] = typer.Option(None, "--height", min=1),
    charset: str = typer.Option("detailed", "--charset"),
    invert: bool = typer.Option(False, "--invert"),
    dither: bool = typer.Option(False, "--dither/--no-dither"),
    fps: Optional[float] = typer.Option(None, "--fps", min=1, max=120),
    duration: Optional[float] = typer.Option(None, "--duration", min=0.01),
    frames: Optional[int] = typer.Option(None, "--frames", min=1),
    backend: str = typer.Option(
        "auto", "--backend", help="Screen backend: auto, mss, or pillow"
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
        fps=fps,
        duration=duration,
        max_frames=frames,
        performance=performance,
        screen_backend=backend,
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
        "glyph", "--mode", "-m", help="glyph, braille, half-block, quadrant"
    ),
    color: str = typer.Option(
        "auto", "--color", "-c", help="auto, none, ansi256, truecolor"
    ),
    width: Optional[int] = typer.Option(None, "--width", "-w", min=1),
    height: Optional[int] = typer.Option(None, "--height", min=1),
    charset: str = typer.Option("detailed", "--charset"),
    invert: bool = typer.Option(False, "--invert"),
    dither: bool = typer.Option(False, "--dither/--no-dither"),
    fps: Optional[float] = typer.Option(None, "--fps", min=1, max=120),
    duration: Optional[float] = typer.Option(None, "--duration", min=0.01),
    frames: Optional[int] = typer.Option(None, "--frames", min=1),
    loop: bool = typer.Option(False, "--loop/--no-loop"),
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
        fps=fps,
        duration=duration,
        max_frames=frames,
        performance=performance,
        loop=loop,
    )


__all__ = ["app", "camera_command", "screen_command", "video_command"]

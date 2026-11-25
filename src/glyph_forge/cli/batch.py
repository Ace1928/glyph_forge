"""Batch scanning and rendering CLI."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

import typer
from rich.console import Console

from ..services.batch import (
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_EXCLUDE_NAMES,
    BatchState,
    iter_video_files,
    process_videos,
)
from ..streaming.naming import VIDEO_EXTS
from .prompting import confirm_action

app = typer.Typer(help="Batch scan and process videos.", rich_markup_mode=None)
console = Console()


def _parse_exts(exts: str | object) -> Set[str]:
    if exts is None:
        return {ext.lower() for ext in VIDEO_EXTS}
    if not isinstance(exts, str):
        exts = str(exts)
    items = [item.strip().lower() for item in exts.split(",") if item.strip()]
    normalized = set()
    for item in items:
        if not item.startswith("."):
            item = "." + item
        normalized.add(item)
    return normalized or {ext.lower() for ext in VIDEO_EXTS}


@app.command("run")
def run(
    roots: List[str] = typer.Argument(["/"], help="Root directories to scan"),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", help="Directory for output files"
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing outputs"),
    resolution: str = typer.Option(
        "720p", "--resolution", "-r", help="Resolution (1080p/720p/480p)"
    ),
    fps: int = typer.Option(30, "--fps", "-f", help="Target FPS"),
    mode: str = typer.Option("gradient", "--mode", "-m", help="Render mode (gradient/braille)"),
    color: str = typer.Option(
        "ansi256", "--color", "-c", help="Color mode (truecolor/ansi256/none)"
    ),
    backend: str = typer.Option("auto", "--backend", help="Frame backend (auto/cv2/ffmpeg)"),
    buffer_seconds: float = typer.Option(
        30.0, "--buffer-seconds", help="Target buffer duration in seconds"
    ),
    prebuffer_seconds: float = typer.Option(
        5.0, "--prebuffer-seconds", help="Minimum prebuffer duration in seconds"
    ),
    max_duration: Optional[float] = typer.Option(
        None, "--duration", "-d", help="Max duration per video (seconds)"
    ),
    mux_audio: bool = typer.Option(
        True, "--mux-audio/--no-mux-audio", help="Mux audio into recorded output"
    ),
    include_exts: str = typer.Option(
        ",".join(sorted({ext.lstrip(".") for ext in VIDEO_EXTS})),
        "--include-exts",
        help="Comma-separated list of video extensions",
    ),
    exclude_dir: List[str] = typer.Option(
        sorted(DEFAULT_EXCLUDE_DIRS),
        "--exclude-dir",
        help="Directories to exclude (repeatable)",
    ),
    exclude_name: List[str] = typer.Option(
        sorted(DEFAULT_EXCLUDE_NAMES),
        "--exclude-name",
        help="Directory names to exclude (repeatable)",
    ),
    follow_links: bool = typer.Option(
        False, "--follow-links", help="Follow symlinks while scanning"
    ),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max number of videos to process"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from previous state"),
    state_dir: Optional[str] = typer.Option(
        None, "--state-dir", help="State directory for resume logs"
    ),
    reset_state: bool = typer.Option(
        False, "--reset-state", help="Clear previous state before running"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Only list matching files, do not process"
    ),
    yes: Optional[bool] = typer.Option(
        None, "--yes/--no", "-y/-n", help="Assume yes/no for prompts"
    ),
):
    """Scan the filesystem for video files and render them in batch."""
    root_list = [str(Path(root)) for root in roots]
    include = _parse_exts(include_exts)
    exclude_dirs = {str(Path(p)) for p in exclude_dir}
    exclude_names = set(exclude_name)

    if not confirm_action(
        f"Batch scan {len(root_list)} root(s) and process matching videos?",
        assume_yes=yes,
        default=False,
    ):
        return

    state_path = Path(state_dir) if state_dir else Path.cwd() / "glyph_forge_output" / "batch_state"
    state = BatchState.load(state_path, reset=reset_state)

    def on_error(err: Exception) -> None:
        console.print(f"[yellow]Scan warning:[/yellow] {err}")

    def on_progress(path: Path, status: str, error: Exception | None) -> None:
        if status == "dry_run":
            console.print(f"[cyan]Found:[/cyan] {path}")
        elif status == "processed":
            console.print(f"[green]Processed:[/green] {path}")
        elif status == "failed":
            console.print(f"[red]Failed:[/red] {path} ({error})")

    videos = iter_video_files(
        root_list,
        include_exts=include,
        exclude_dirs=exclude_dirs,
        exclude_names=exclude_names,
        follow_links=follow_links,
        on_error=on_error,
    )

    processed, failed = process_videos(
        videos,
        output_dir=output_dir,
        overwrite=overwrite,
        resolution=resolution,
        fps=fps,
        mode=mode,
        color=color,
        backend=backend,
        buffer_seconds=buffer_seconds,
        prebuffer_seconds=prebuffer_seconds,
        max_duration=max_duration,
        mux_audio=mux_audio,
        state=state,
        resume=resume,
        limit=limit,
        dry_run=dry_run,
        on_progress=on_progress,
    )

    console.print(
        f"[bold cyan]Batch complete.[/bold cyan] Processed: {processed} | Failed: {failed} | State: {state_path}"
    )

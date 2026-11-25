"""Audio utilities for Glyph Forge."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ..streaming.audio_tools import mux_audio
from ..streaming.naming import build_metadata, write_metadata
from .prompting import confirm_action

app = typer.Typer(help="Audio tools (mux/sync)", rich_markup_mode=None)


@app.command("mux")
def mux(
    video: str = typer.Argument(..., help="Video file to mux audio into"),
    audio: Optional[str] = typer.Option(None, "--audio", help="Audio file path"),
    youtube: Optional[str] = typer.Option(None, "--youtube", help="YouTube URL to fetch audio"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output video path"),
    offset: float = typer.Option(0.0, "--offset", help="Audio offset in seconds"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing output"),
    metadata: bool = typer.Option(True, "--metadata/--no-metadata", help="Write metadata sidecar"),
    yes: Optional[bool] = typer.Option(
        None, "--yes/--no", "-y/-n", help="Assume yes/no for prompts"
    ),
):
    """Mux audio into a video file with optional YouTube download."""
    if (audio is None) == (youtube is None):
        raise typer.BadParameter("Provide exactly one of --audio or --youtube")

    audio_source = youtube or audio
    if youtube and not confirm_action(
        "Download audio from YouTube URL?",
        assume_yes=yes,
        default=True,
    ):
        return

    output_path = Path(output) if output else None
    try:
        result = mux_audio(
            Path(video),
            audio_source,
            output_path=output_path,
            offset_seconds=offset,
            overwrite=overwrite,
        )
    except FileExistsError:
        if not confirm_action(
            "Output exists. Overwrite?",
            assume_yes=yes,
            default=False,
        ):
            return
        result = mux_audio(
            Path(video),
            audio_source,
            output_path=output_path,
            offset_seconds=offset,
            overwrite=True,
        )
    except Exception as exc:
        raise typer.Exit(str(exc))

    if not result:
        raise typer.Exit("Audio mux failed")

    typer.echo(f"Muxed audio saved to {result}")
    if metadata:
        meta = build_metadata(
            source=audio_source or str(video),
            output_path=result,
            title=Path(result).stem,
            info=None,
            extra={
                "video_source": str(video),
                "audio_source": audio_source,
                "offset_seconds": offset,
                "operation": "audio_mux",
            },
        )
        write_metadata(meta, result)

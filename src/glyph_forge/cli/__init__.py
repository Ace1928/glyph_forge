"""Unified command-line experience for Glyph Forge."""

from __future__ import annotations

import json
import logging
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..config.settings import ConfigManager, get_config
from ..runtime import (
    configure_utf8_stdio,
    detect_runtime_profile,
    reexec_clean_android_environment,
    runtime_report,
)
from .live import app as live_app
from .live import camera_command, screen_command, source_command
from .plugins import app as plugins_app

logger = logging.getLogger(__name__)
console = Console()
error_console = Console(stderr=True)

app = typer.Typer(
    name="glyph-forge",
    help="Turn images, text, and video into expressive glyph art.",
    add_completion=True,
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(live_app, name="live")
app.add_typer(plugins_app, name="plugins")
app.command("webcam")(camera_command)
app.command("desktop")(screen_command)
app.command("stream")(source_command)


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context) -> None:
    """Launch Glyph Forge or choose a focused command."""

    if ctx.invoked_subcommand is None:
        display_banner()
        _display_quick_start()


@app.command()
def version(
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON."
    ),
) -> None:
    """Show Glyph Forge and environment versions."""

    from .. import __version__

    data = {
        "glyph_forge": __version__,
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    if json_output:
        typer.echo(json.dumps(data, sort_keys=True))
        return

    table = Table(show_header=False, box=None)
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="yellow")
    table.add_row("Glyph Forge", data["glyph_forge"])
    table.add_row("Python", data["python"])
    table.add_row("Platform", data["platform"])
    console.print(Panel(table, title="Glyph Forge", border_style="bright_yellow"))


@app.command("image")
def image_command(
    source: Optional[Path] = typer.Argument(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Image to transform.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Save the glyph art to this file.",
    ),
    width: Optional[int] = typer.Option(
        None,
        "--width",
        "-w",
        min=1,
        help="Target width in columns; adaptive default when omitted.",
    ),
    height: Optional[int] = typer.Option(
        None,
        "--height",
        min=1,
        help="Target height in rows; computed from the aspect ratio when omitted.",
    ),
    charset: str = typer.Option(
        "general",
        "--charset",
        "-c",
        help="Character set for brightness mapping.",
    ),
    style: Optional[str] = typer.Option(
        None,
        "--style",
        "-s",
        help="Style preset to apply to the rendered art.",
    ),
    color: str = typer.Option("none", "--color", help="none, ansi, or html"),
    render_mode: str = typer.Option(
        "glyph",
        "--mode",
        "-m",
        help="glyph, edge, braille, half-block, or quadrant",
    ),
    edge_algorithm: str = typer.Option(
        "sobel",
        "--edge-algorithm",
        help="sobel, prewitt, scharr, laplacian, or canny",
    ),
    edge_threshold: int = typer.Option(
        48,
        "--edge-threshold",
        min=0,
        max=255,
        help="Edge detector sensitivity (0–255).",
    ),
    aspect: Optional[float] = typer.Option(
        None,
        "--aspect",
        min=0.01,
        help="Force character-grid width/height ratio.",
    ),
    invert: bool = typer.Option(
        False, "--invert", help="Invert light and dark glyph mapping."
    ),
    brightness: float = typer.Option(
        1.0, "--brightness", min=0.0, max=2.0, help="Brightness multiplier (0.0–2.0)."
    ),
    contrast: float = typer.Option(
        1.0, "--contrast", min=0.0, max=2.0, help="Contrast multiplier (0.0–2.0)."
    ),
    optimize: bool = typer.Option(
        False,
        "--optimize",
        help="Automatically stretch source contrast before rendering.",
    ),
    dithering: bool = typer.Option(
        False, "--dither/--no-dither", help="Apply dithering to smooth gradients."
    ),
    fit_terminal: bool = typer.Option(
        True, "--fit/--no-fit", help="Fit the output to the terminal width."
    ),
    preview: bool = typer.Option(
        True, "--preview/--no-preview", help="Print the result in the terminal."
    ),
    performance: str = typer.Option(
        "auto",
        "--performance",
        help="auto, eco, balanced, or workstation",
    ),
    list_charsets: bool = typer.Option(
        False,
        "--list-charsets",
        help="List every installed character set and exit.",
    ),
    preview_charset: Optional[str] = typer.Option(
        None,
        "--preview-charset",
        help="Preview one character set and exit.",
    ),
    sample: Optional[Path] = typer.Option(
        None,
        "--sample",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Optional image used by --preview-charset.",
    ),
) -> None:
    """Convert an image with sensible adaptive defaults and an instant preview."""

    from ..live.renderers import RenderMode, normalize_render_mode
    from ..plugins import PluginError
    from ..services.image_to_glyph import ImageGlyphConverter
    from ..utils.alphabet_manager import AlphabetManager

    if list_charsets:
        table = Table(title="Character sets")
        table.add_column("Name", style="bold cyan")
        table.add_column("Glyphs")
        for name in sorted(AlphabetManager.list_available_alphabets()):
            glyphs = AlphabetManager.get_alphabet(name)
            table.add_row(name, glyphs[:48] + ("…" if len(glyphs) > 48 else ""))
        console.print(table)
        return
    if preview_charset is not None:
        if preview_charset not in AlphabetManager.list_available_alphabets():
            raise typer.BadParameter(
                f"Unknown character set {preview_charset!r}",
                param_hint="--preview-charset",
            )
        glyphs = AlphabetManager.get_alphabet(preview_charset)
        console.print(f"[bold cyan]{preview_charset}[/bold cyan]\n{glyphs}")
        if sample is not None:
            typer.echo(
                ImageGlyphConverter(
                    charset=preview_charset,
                    width=width or 80,
                    height=height,
                    auto_scale=fit_terminal,
                ).convert(str(sample))
            )
        return
    if sample is not None:
        raise typer.BadParameter(
            "--sample requires --preview-charset", param_hint="--sample"
        )
    if source is None:
        raise typer.BadParameter(
            "An image path is required unless listing or previewing character sets",
            param_hint="source",
        )

    color_mode = color.casefold()
    if color_mode not in {"none", "ansi", "html"}:
        raise typer.BadParameter("Choose none, ansi, or html", param_hint="--color")
    try:
        selected_render_mode = normalize_render_mode(render_mode)
    except (PluginError, ValueError) as exc:
        choices = ", ".join(item.value for item in RenderMode)
        raise typer.BadParameter(
            f"Choose {choices}, or plugin:plugin-id/renderer ({exc})",
            param_hint="--mode",
        ) from exc
    if selected_render_mode is not RenderMode.GLYPH and color_mode == "html":
        raise typer.BadParameter(
            "HTML colour is available in glyph mode; use --color none or ansi",
            param_hint="--color",
        )
    try:
        profile = detect_runtime_profile(performance)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--performance") from exc

    selected_width = width or profile.image_width
    selected_height = height
    if aspect is not None and selected_height is None:
        selected_height = max(1, round(selected_width / aspect))
    destination = str(output) if output is not None else None
    prepared_source: str | Any = str(source)
    if optimize:
        from PIL import Image, ImageOps

        with Image.open(source) as image:
            prepared_source = ImageOps.autocontrast(image.convert("RGB"))
    if selected_render_mode is RenderMode.GLYPH:
        converter = ImageGlyphConverter(
            charset=charset,
            width=selected_width,
            height=selected_height,
            invert=invert,
            brightness=brightness,
            contrast=contrast,
            auto_scale=fit_terminal,
            dithering=dithering,
            threads=profile.workers,
        )
        if color_mode == "none":
            result = converter.convert(
                prepared_source, output_path=destination, style=style
            )
        else:
            result = converter.convert_color(
                prepared_source, output_path=destination, color_mode=color_mode
            )
    else:
        import numpy as np
        from PIL import Image, ImageEnhance

        from ..core.style_manager import apply_style
        from ..live.renderers import FrameRenderer, RenderConfig

        if fit_terminal:
            selected_width = min(
                selected_width,
                max(20, shutil.get_terminal_size((selected_width, 24)).columns - 2),
            )
        if isinstance(prepared_source, Image.Image):
            prepared = prepared_source
        else:
            with Image.open(source) as image:
                prepared = image.convert("RGB")
        if not isinstance(prepared_source, Image.Image):
            prepared = prepared.copy()
        if brightness != 1:
            prepared = ImageEnhance.Brightness(prepared).enhance(brightness)
        if contrast != 1:
            prepared = ImageEnhance.Contrast(prepared).enhance(contrast)
        pixels = np.asarray(prepared, dtype=np.uint8)
        try:
            result = (
                FrameRenderer(
                    RenderConfig(
                        width=selected_width,
                        height=selected_height,
                        mode=selected_render_mode,
                        color="truecolor" if color_mode == "ansi" else "none",
                        charset=charset,
                        invert=invert,
                        dither=dithering,
                        edge_algorithm=edge_algorithm,
                        edge_threshold=edge_threshold,
                        resample=profile.resample,
                    )
                )
                .render(pixels)
                .text
            )
        except PluginError as exc:
            error_console.print(f"[bold red]Plugin render failed:[/bold red] {exc}")
            raise typer.Exit(2) from exc
        if style:
            result = apply_style(result, style_name=style)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result, encoding="utf-8")
    if result.startswith("Error"):
        console.print(f"[bold red]{result}[/bold red]")
        raise typer.Exit(1)
    if preview or output is None:
        typer.echo(result)
    if output is not None:
        error_console.print(f"[green]Saved[/green] {output}")


@app.command("text")
def text_command(
    text: Optional[str] = typer.Argument(None, help="Text to turn into a banner."),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Save the banner to this file.",
    ),
    font: str = typer.Option("slant", "--font", "-f", help="FIGlet font name or path."),
    style: str = typer.Option("minimal", "--style", "-s", help="Style preset name."),
    width: int = typer.Option(
        80, "--width", "-w", min=10, help="Maximum banner width in columns."
    ),
    color: bool = typer.Option(
        False, "--color/--no-color", help="Add ANSI colour to the banner."
    ),
    list_fonts: bool = typer.Option(
        False,
        "--list-fonts",
        help="List every installed FIGlet font and exit.",
    ),
    list_styles: bool = typer.Option(
        False,
        "--list-styles",
        help="List every text style and exit.",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Label the result as a preview (retained for compatibility).",
    ),
) -> None:
    """Generate a styled text banner."""

    from ..api import get_api
    from ..services.text_to_banner import text_to_banner

    if list_fonts:
        fonts = get_api().get_available_fonts()
        console.print(Columns(sorted(fonts), equal=True, expand=True))
        return
    if list_styles:
        table = Table(title="Text styles")
        table.add_column("Name", style="bold cyan")
        table.add_column("Description")
        for name, data in sorted(get_api().get_available_styles().items()):
            table.add_row(name, str(data.get("description", "")))
        console.print(table)
        return
    if text is None:
        raise typer.BadParameter(
            "Text is required unless --list-fonts or --list-styles is used",
            param_hint="text",
        )

    result = text_to_banner(text, style=style, font=font, width=width, color=color)
    if preview:
        error_console.print(f"[cyan]Preview[/cyan] · {font} · {style}")
    typer.echo(result)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result, encoding="utf-8")
        error_console.print(f"[green]Saved[/green] {output}")


# Compatibility names now point directly at the maintained workflows instead
# of registering two duplicate command trees.
app.command("imagize", hidden=True)(image_command)
app.command("bannerize", hidden=True)(text_command)


@app.command("video")
def video_command(
    source: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Video to render as full-colour glyph art.",
    ),
    output: Optional[Path] = typer.Argument(
        None,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Output video (defaults to <input>.glyph.mp4).",
    ),
    width: Optional[int] = typer.Option(
        None, "--width", min=2, help="Output video width in pixels."
    ),
    height: Optional[int] = typer.Option(
        None, "--height", min=2, help="Output video height in pixels."
    ),
    columns: Optional[int] = typer.Option(
        None, "--columns", min=1, help="Glyph grid columns."
    ),
    rows: Optional[int] = typer.Option(None, "--rows", min=1, help="Glyph grid rows."),
    charset: str = typer.Option(
        "detailed", "--charset", "-c", help="Character set for brightness mapping."
    ),
    font: Optional[str] = typer.Option(
        None,
        "--font",
        help="Monospace font path/name; auto-detected when omitted.",
    ),
    start: float = typer.Option(0.0, "--start", min=0.0, help="Start time in seconds."),
    duration: Optional[float] = typer.Option(
        None,
        "--duration",
        min=0.001,
        help="Seconds to render; the rest of the video is used when omitted.",
    ),
    crf: int = typer.Option(
        18,
        "--crf",
        min=0,
        max=51,
        help="FFmpeg quality; lower values are higher quality (0–51).",
    ),
    preset: str = typer.Option("veryfast", "--preset", help="FFmpeg x264 preset."),
    ffmpeg: str = typer.Option("ffmpeg", "--ffmpeg", help="FFmpeg executable."),
    workers: Optional[int] = typer.Option(
        None,
        "--workers",
        min=1,
        max=64,
        help="Ordered render workers; adaptive by default.",
    ),
    performance: str = typer.Option(
        "auto",
        "--performance",
        help="auto, eco, balanced, or workstation",
    ),
    progress_output: bool = typer.Option(
        True,
        "--progress/--quiet",
        help="Show periodic streaming progress.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit complete output and performance metrics as JSON.",
    ),
) -> None:
    """Stream a full-colour glyph MP4 with the source audio preserved."""

    from ..live.video import (
        VideoExportConfig,
        VideoExportError,
        VideoExportProgress,
        export_glyph_video,
        with_video_overrides,
    )

    destination = output or source.with_name(f"{source.stem}.glyph.mp4")
    try:
        config = with_video_overrides(
            VideoExportConfig.adaptive(performance),
            width=width,
            height=height,
            columns=columns,
            rows=rows,
            charset=charset,
            font=font,
            start=start,
            duration=duration,
            crf=crf,
            preset=preset,
            ffmpeg=ffmpeg,
            workers=workers,
        ).validated()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if not json_output:
        error_console.print(
            f"Rendering [cyan]{config.columns}×{config.rows}[/cyan] glyphs to "
            f"[cyan]{config.width}×{config.height}[/cyan] with "
            f"[cyan]{config.workers}[/cyan] ordered worker"
            f"{'s' if config.workers != 1 else ''}"
        )
    last_report = -10.0

    def report_progress(progress: VideoExportProgress) -> None:
        nonlocal last_report
        complete = progress.total_frames == progress.rendered_frames
        if (
            progress.rendered_frames != 1
            and progress.elapsed - last_report < 5
            and not complete
        ):
            return
        last_report = progress.elapsed
        total = str(progress.total_frames) if progress.total_frames else "?"
        error_console.print(
            f"  {progress.rendered_frames}/{total} frames ({progress.elapsed:.1f}s)"
        )

    try:
        result = export_glyph_video(
            source,
            destination,
            config,
            progress=report_progress if progress_output and not json_output else None,
        )
    except VideoExportError as exc:
        error_console.print(f"[bold red]Video export failed:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    if json_output:
        typer.echo(json.dumps(result.to_dict(), sort_keys=True))
        return
    error_console.print(
        f"[green]Saved[/green] {result.output} · {result.rendered_frames} frames "
        f"in {result.elapsed:.1f}s · {result.render_fps:.2f} render FPS · "
        f"{result.realtime_factor:.2f}× real-time · "
        f"{result.output_bytes / (1024**2):.1f} MiB"
    )


@app.command()
def styles(
    preview: bool = typer.Option(False, "--preview", help="Render a short sample."),
) -> None:
    """Browse image charsets and text styles."""

    from ..core.style_manager import get_available_styles
    from ..utils.alphabet_manager import AlphabetManager

    table = Table(title="Glyph Forge styles")
    table.add_column("Kind", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Preview")
    for name in sorted(get_available_styles()):
        table.add_row("text", name, "Glyph Forge" if preview else "")
    for name in sorted(AlphabetManager.list_available_alphabets()):
        sample = AlphabetManager.get_alphabet(name)[:32] if preview else ""
        table.add_row("image", name, sample)
    console.print(table)


@app.command()
def doctor(
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON."
    ),
    performance: str = typer.Option(
        "auto",
        "--performance",
        help="Inspect a specific performance mode.",
    ),
) -> None:
    """Check portability, optional features, tools, and adaptive defaults."""

    try:
        report = runtime_report(performance)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--performance") from exc
    # The source version is authoritative in editable/development checkouts.
    from .. import __version__

    report["glyph_forge"] = __version__
    from ..plugins import (
        PLUGIN_API_VERSION,
        get_plugin_registry,
        plugins_enabled,
    )

    report["plugins"] = {
        "api_version": PLUGIN_API_VERSION,
        "enabled": plugins_enabled(),
        "installed": [
            item.to_dict() for item in get_plugin_registry().inventory(load=False)
        ],
    }
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return

    profile = report["profile"]
    summary = Table(show_header=False, box=None)
    summary.add_column("Property", style="bold cyan")
    summary.add_column("Value")
    summary.add_row("Glyph Forge", report["glyph_forge"])
    summary.add_row("Python", f"{report['python']} ({report['implementation']})")
    summary.add_row("Platform", f"{report['system']} / {report['machine']}")
    summary.add_row("Adaptive profile", profile["tier"])
    summary.add_row("CPU / workers", f"{profile['cpu_count']} / {profile['workers']}")
    memory = profile["memory_bytes"]
    summary.add_row("Memory", _format_bytes(memory) if memory else "unknown")
    summary.add_row(
        "Image / stream width", f"{profile['image_width']} / {profile['stream_width']}"
    )
    summary.add_row("Target FPS", str(profile["target_fps"]))
    console.print(Panel(summary, title="Runtime", border_style="cyan"))

    features = Table(title="Features")
    features.add_column("Status")
    features.add_column("Feature")
    features.add_column("Purpose")
    features.add_column("Install")
    for item in report["capabilities"]:
        status = (
            "[green]ready[/green]" if item["available"] else "[yellow]optional[/yellow]"
        )
        features.add_row(
            status,
            item["label"],
            item["purpose"]
            + (
                f" ({item['detail']})"
                if item.get("detail") and not item["available"]
                else ""
            ),
            "" if item["available"] else (item["install_hint"] or ""),
        )
    console.print(features)

    plugin_report = report["plugins"]
    installed_plugins = plugin_report["installed"]
    plugins = Table(title=f"Plugins · API {plugin_report['api_version']}")
    plugins.add_column("State")
    plugins.add_column("Identifier", style="bold cyan")
    plugins.add_column("Distribution")
    plugins.add_column("Entry point")
    for item in installed_plugins:
        plugins.add_row(
            str(item["state"]),
            str(item["identifier"]),
            str(item["distribution"] or "—"),
            str(item["entry_point"] or item["error"] or "—"),
        )
    if installed_plugins:
        console.print(plugins)
    else:
        status = "enabled; none installed" if plugin_report["enabled"] else "disabled"
        console.print(
            f"[dim]Plugins · API {plugin_report['api_version']} · {status}[/dim]"
        )


@app.command()
def launch(
    interface: str = typer.Argument(
        "auto", help="Interface to launch: auto, gui, tui, or cli."
    ),
) -> None:
    """Launch the best available interface or an explicitly selected one."""

    selected = interface.casefold()
    if selected not in {"auto", "gui", "studio", "tui", "cli"}:
        raise typer.BadParameter(
            "Choose auto, gui, studio, tui, or cli", param_hint="interface"
        )
    dependencies = check_for_external_dependencies()
    if selected == "auto":
        selected = "tui" if dependencies["textual"] and sys.stdin.isatty() else "cli"
    if selected == "tui":
        interactive()
        return
    if selected in {"gui", "studio"}:
        studio(
            host="127.0.0.1",
            port=0,
            open_browser=True,
            allow_network=False,
            lan=False,
            share_links=False,
            advertise_host=None,
            share_ttl=3600,
            duration=None,
        )
        return
    display_banner()
    _display_quick_start()


@app.command()
def interactive() -> None:
    """Launch the optional full-screen terminal interface."""

    try:
        from ..ui.tui import GlyphForgeApp
    except ImportError as exc:
        if exc.name and exc.name.startswith("textual"):
            console.print("[yellow]The TUI is optional.[/yellow]")
            console.print(
                "Install it with: [bold]pip install 'glyph-forge[tui]'[/bold]"
            )
            raise typer.Exit(2) from exc
        raise
    result = GlyphForgeApp().run()
    if result == "studio":
        studio(
            host="127.0.0.1",
            port=0,
            open_browser=True,
            allow_network=False,
            lan=False,
            share_links=False,
            advertise_host=None,
            share_ttl=3600,
            duration=None,
        )


@app.command()
def studio(
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Address for the local studio server."
    ),
    port: int = typer.Option(
        0, "--port", min=0, max=65535, help="Port; zero chooses a free port."
    ),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the studio in the default browser."
    ),
    allow_network: bool = typer.Option(
        False,
        "--allow-network",
        help="Explicitly allow a non-loopback bind for trusted LAN sharing.",
    ),
    lan: bool = typer.Option(
        False,
        "--lan",
        help="Listen on the trusted LAN and enable temporary share links.",
    ),
    share_links: bool = typer.Option(
        False,
        "--share-links",
        help="Enable bounded temporary links from the Studio export panel.",
    ),
    advertise_host: Optional[str] = typer.Option(
        None,
        "--advertise-host",
        help="Hostname or IP embedded in links (useful with VPNs or many NICs).",
    ),
    share_ttl: int = typer.Option(
        3600,
        "--share-ttl",
        min=1,
        max=86400,
        help="Lifetime of temporary links in seconds (maximum one day).",
    ),
    duration: Optional[float] = typer.Option(
        None,
        "--duration",
        min=0.01,
        hidden=True,
        help="Stop automatically after N seconds (automation/testing).",
    ),
) -> None:
    """Open the private local GUI for files, video, webcam, and screen art."""

    from ..studio import StudioError, StudioServer

    if lan:
        if host == "127.0.0.1":
            host = "0.0.0.0"
        allow_network = True
        share_links = True
    try:
        server = StudioServer(
            host,
            port,
            allow_network=allow_network,
            quiet=True,
            share_links=share_links,
            advertise_host=advertise_host,
            share_ttl=float(share_ttl),
        ).start(open_browser=open_browser)
    except (StudioError, ValueError) as exc:
        error_console.print(f"[bold red]Could not start studio:[/bold red] {exc}")
        raise typer.Exit(2) from exc

    console.print(Panel.fit(f"[bold cyan]{server.url}[/bold cyan]", title="Studio"))
    if server.sharing_enabled:
        error_console.print(
            f"Temporary link sharing is enabled at {server.public_url} "
            f"for up to {share_ttl} seconds."
        )
        error_console.print("Use it only on a trusted LAN; links are unencrypted HTTP.")
    else:
        error_console.print(
            "Your media stays in this browser session. Ctrl+C stops it."
        )
    try:
        if duration is not None:
            threading.Event().wait(duration)
        else:
            server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


@app.command()
def share(
    source: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Rendered image, video, audio, text, or other file to share.",
    ),
    lan: bool = typer.Option(
        False,
        "--lan",
        help="Listen beyond this device on a trusted local network.",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Bind address; --lan changes the default to 0.0.0.0.",
    ),
    port: int = typer.Option(
        0, "--port", min=0, max=65535, help="Port; zero chooses a free port."
    ),
    advertise_host: Optional[str] = typer.Option(
        None,
        "--advertise-host",
        help="Hostname or IP friends should use when automatic detection is wrong.",
    ),
    ttl: int = typer.Option(
        3600,
        "--ttl",
        min=1,
        max=86400,
        help="Link lifetime in seconds (maximum one day).",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open/--no-open",
        help="Open the temporary link in the default browser.",
    ),
    duration: Optional[float] = typer.Option(
        None,
        "--duration",
        min=0.01,
        hidden=True,
        help="Stop earlier after N seconds (automation/testing).",
    ),
) -> None:
    """Create a seekable, temporary link to exactly one local file."""

    import webbrowser

    from ..studio import StudioError, StudioServer

    if lan and host == "127.0.0.1":
        host = "0.0.0.0"
    server: StudioServer | None = None
    try:
        server = StudioServer(
            host,
            port,
            allow_network=lan,
            quiet=True,
            share_links=True,
            advertise_host=advertise_host,
            share_ttl=float(ttl),
            browser_shares=False,
        )
        publication = server.publish_file(source, ttl=float(ttl))
        server.start()
    except (StudioError, ValueError) as exc:
        if server is not None:
            server.close()
        error_console.print(f"[bold red]Could not share file:[/bold red] {exc}")
        raise typer.Exit(2) from exc

    details = Table(show_header=False, box=None)
    details.add_column("Property", style="bold cyan")
    details.add_column("Value")
    details.add_row("Link", Text(publication.url, style="bright_yellow"))
    details.add_row("File", Text(publication.filename))
    details.add_row("Size", _format_bytes(publication.size))
    details.add_row("Expires", f"in {ttl} seconds")
    console.print(Panel.fit(details, title="Temporary share"))
    error_console.print(
        "The file is streamed in place with video seeking; it is not copied or uploaded."
    )
    if lan:
        error_console.print(
            "Anyone with this unencrypted link on the trusted LAN can access it until "
            "the command stops."
        )
    else:
        error_console.print(
            "This loopback link works only on this device. Add --lan to share nearby."
        )
    if open_browser:
        webbrowser.open(publication.url, new=2)
    try:
        lifetime = float(ttl) if duration is None else min(float(ttl), duration)
        threading.Event().wait(lifetime)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


@app.command()
def demo(
    mode: str = typer.Option(
        "all",
        "--mode",
        "-m",
        help="all, or one of glyph, edge, braille, half-block, quadrant.",
    ),
    width: Optional[int] = typer.Option(
        None,
        "--width",
        "-w",
        min=10,
        help="Target width in columns; fitted to the terminal when omitted.",
    ),
    color: bool = typer.Option(
        True,
        "--color/--no-color",
        help="Add ANSI colour when the terminal supports it.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save the complete showcase text to this file."
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        help="Save every scene, render, meme, and thumbnail into this directory.",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Never touch the network; craft every asset locally instead.",
    ),
    media: bool = typer.Option(
        True,
        "--media/--no-media",
        help="Fetch popular video thumbnails for the video scene.",
    ),
    performance: str = typer.Option(
        "auto", "--performance", help="auto, eco, balanced, or workstation."
    ),
) -> None:
    """Run the self-contained showcase: memes, media, modes, fonts, styles."""

    from ..demo import run_demo

    try:
        result = run_demo(
            mode=mode,
            width=width,
            color=color,
            offline=offline,
            media=media,
            performance=performance,
            output_dir=output_dir,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        typer.echo(result.text)
    except UnicodeEncodeError:
        typer.echo(result.text.encode("utf-8", "replace").decode("utf-8", "replace"))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.text, encoding="utf-8")
        error_console.print(f"[green]Saved[/green] {output}")
    if output_dir is not None:
        error_console.print(
            f"[green]Saved[/green] {len(result.artifacts)} artifacts to {output_dir}"
        )
        for artifact in result.artifacts:
            error_console.print(f"  {artifact.name}")
    stats = result.stats
    error_console.print(
        f"[cyan]Show complete[/cyan] · {stats.renders} renders · "
        f"{stats.assets_fetched} assets fetched · "
        f"{stats.assets_fallback} crafted offline · {stats.elapsed_seconds:.1f}s"
    )


@app.command()
def benchmark(
    mode: str = typer.Option(
        "all",
        "--mode",
        "-m",
        help="all, a built-in mode, or plugin:plugin-id/renderer",
    ),
    iterations: int = typer.Option(
        3,
        "--iterations",
        "-n",
        min=1,
        max=100,
        help="Timed rendering passes per mode.",
    ),
    warmup: int = typer.Option(
        1, "--warmup", min=0, max=20, help="Unmeasured warm-up passes before timing."
    ),
    performance: str = typer.Option(
        "auto", "--performance", help="auto, eco, balanced, or workstation."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Measure renderer throughput with a deterministic local frame."""

    from ..benchmark import benchmark_renderers
    from ..live.renderers import PluginRenderMode, RenderMode, normalize_render_mode
    from ..plugins import PluginError

    if mode.casefold() == "all":
        modes: list[RenderMode | PluginRenderMode | str] = list(RenderMode)
    else:
        try:
            modes = [normalize_render_mode(mode)]
        except (PluginError, ValueError) as exc:
            raise typer.BadParameter(
                "Choose all, a built-in mode, or plugin:plugin-id/renderer",
                param_hint="--mode",
            ) from exc
    try:
        results = benchmark_renderers(
            performance,
            modes=modes,
            iterations=iterations,
            warmup=warmup,
        )
    except (PluginError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if as_json:
        typer.echo(json.dumps([item.to_dict() for item in results], indent=2))
        return
    table = Table(title=f"Renderer benchmark · {performance}")
    table.add_column("Mode", style="bold cyan")
    table.add_column("Source")
    table.add_column("Grid")
    table.add_column("Latency", justify="right")
    table.add_column("Throughput", justify="right")
    for item in results:
        table.add_row(
            item.mode,
            f"{item.source_width}×{item.source_height}",
            f"{item.columns}×{item.rows}",
            f"{item.milliseconds:.2f} ms",
            f"{item.frames_per_second:.1f} FPS",
        )
    console.print(table)


@app.command("list-commands")
def list_commands() -> None:
    """Show the main workflows and compatibility commands."""

    table = Table(title="Glyph Forge commands")
    table.add_column("Command", style="bold cyan")
    table.add_column("Use")
    for command, description in (
        ("image", "Convert and preview an image"),
        ("text", "Create a text banner"),
        ("video", "Stream a full-colour glyph video to MP4"),
        ("live", "View a camera, video, or desktop with bounded latency"),
        ("studio", "Open the local browser GUI and sharing surface"),
        ("share", "Create a temporary seekable link to one local file"),
        ("demo", "Render a built-in showcase with no input file"),
        ("benchmark", "Measure adaptive renderer throughput"),
        ("plugins", "Discover and diagnose third-party extensions"),
        ("webcam", "Direct alias for live camera"),
        ("desktop", "Direct alias for live screen"),
        ("stream", "Open any built-in or plugin live source"),
        ("styles", "Browse charsets and styles"),
        ("launch", "Choose CLI or TUI automatically"),
        ("doctor", "Inspect features and adaptive defaults"),
        ("imagize", "Compatible image command alias"),
        ("bannerize", "Compatible banner command alias"),
    ):
        table.add_row(command, description)
    console.print(table)


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


def display_banner() -> None:
    """Display a compact banner that works in narrow and colorless terminals."""

    title = Text("GLYPH FORGE", style="bold bright_yellow", justify="center")
    subtitle = Text(
        "Images · text · video → expressive character art",
        style="cyan",
        justify="center",
    )
    console.print(
        Panel(Text.assemble(title, "\n", subtitle), border_style="bright_yellow")
    )


def _display_quick_start() -> None:
    console.print("[bold]Quick start[/bold]")
    console.print("  glyph-forge image photo.jpg")
    console.print("  glyph-forge text 'Hello friends'")
    console.print("  glyph-forge video clip.mp4")
    console.print("  glyph-forge webcam")
    console.print("  glyph-forge desktop")
    console.print("  glyph-forge studio")
    console.print("  glyph-forge share render.mp4 --lan")
    console.print("  glyph-forge launch tui")
    console.print("  glyph-forge doctor")
    console.print("\nRun [cyan]glyph-forge --help[/cyan] for every option.")


def check_for_external_dependencies() -> Dict[str, bool]:
    """Return compatibility feature flags used by callers and launch routing."""

    capabilities = {
        item["key"]: bool(item["available"])
        for item in runtime_report()["capabilities"]
    }
    return {
        "textual": capabilities.get("textual", False),
        "pillow": capabilities.get("PIL", False),
        "numpy": capabilities.get("numpy", False),
        "opencv": capabilities.get("cv2", False),
        "mss": capabilities.get("mss", False),
        "ffmpeg": capabilities.get("ffmpeg", False),
    }


def get_settings() -> Union[Dict[str, Any], ConfigManager]:
    """Load persisted settings with a small fallback for read-only systems."""

    try:
        return get_config()
    except (OSError, ValueError) as exc:
        logger.warning("Could not load settings: %s", exc)
        return {
            "banner": {"default_font": "slant", "default_width": 80},
            "image": {"default_charset": "general", "default_width": 100},
            "io": {"color_output": True},
        }


def main() -> None:
    """Console-script entry point."""

    reexec_clean_android_environment()
    configure_utf8_stdio()
    app()


if __name__ == "__main__":
    main()

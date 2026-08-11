"""Project, preset, and bounded batch command surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..batch import BatchError, BatchProgress, items_for_sources, render_batch
from ..contracts import GlyphForgeRenderError, RenderRequest
from ..projects import (
    PRESET_SUFFIX,
    PROJECT_SUFFIX,
    GlyphProject,
    ProjectError,
    ProjectSession,
    RecentProjectStore,
    RenderPreset,
    RenderVariant,
    create_portable_project,
    load_preset,
    load_project,
    recovery_path,
    save_preset,
)
from ..rendering import render_image

console = Console()
error_console = Console(stderr=True)

project_app = typer.Typer(
    name="project",
    help="Create, recover, render, and manage portable Glyph Forge projects.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
preset_app = typer.Typer(
    name="preset",
    help="Create and exchange render settings across every interface.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _variant(project: GlyphProject, identifier: str | None) -> RenderVariant:
    if identifier is None:
        return project.active
    selected = identifier.casefold()
    for variant in project.variants:
        if variant.identifier == selected:
            return variant
    choices = ", ".join(item.identifier for item in project.variants)
    raise typer.BadParameter(
        f"Unknown variant {identifier!r}; choose {choices}",
        param_hint="--variant",
    )


def _remember_project(path: Path) -> None:
    """Update optional recent history without changing command success."""

    try:
        RecentProjectStore().touch(path)
    except ProjectError as exc:
        error_console.print(
            f"[yellow]Project succeeded, but recent history was not updated:[/yellow] {exc}"
        )


@project_app.command("new")
def new_project_command(
    project_path: Path = typer.Argument(
        ...,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help=f"Project document (recommended suffix {PROJECT_SUFFIX}).",
    ),
    source: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Source asset to include or reference.",
    ),
    name: Optional[str] = typer.Option(None, "--name", help="Friendly project name."),
    preset: Optional[Path] = typer.Option(
        None,
        "--preset",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Start with a shared render preset.",
    ),
    copy_asset: bool = typer.Option(
        True,
        "--copy-asset/--reference-only",
        help="Copy external media into a portable assets folder by default.",
    ),
) -> None:
    """Create a portable project, safely copying an external source by default."""

    if project_path.exists():
        raise typer.BadParameter("Project already exists", param_hint="project_path")
    try:
        request = load_preset(preset).request if preset is not None else RenderRequest()
        project = create_portable_project(
            project_path,
            source,
            name=name,
            request=request,
            copy_external=copy_asset,
        )
        _remember_project(project_path)
    except ProjectError as exc:
        error_console.print(f"[bold red]Could not create project:[/bold red] {exc}")
        raise typer.Exit(2) from exc
    error_console.print(
        f"[green]Created[/green] {project_path} · source {project.source.path} · "
        f"{request.width} columns"
    )


@project_app.command("info")
def project_info_command(
    project_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON."
    ),
) -> None:
    """Inspect variants, source portability, and recovery state."""

    try:
        project = load_project(project_path)
        asset = project.source.resolve(project_path)
        _remember_project(project_path)
    except ProjectError as exc:
        error_console.print(f"[bold red]Could not open project:[/bold red] {exc}")
        raise typer.Exit(2) from exc
    values = project.to_dict() | {
        "project_path": str(project_path),
        "resolved_source": str(asset),
        "source_available": asset.is_file(),
        "recovery_available": recovery_path(project_path).is_file(),
    }
    if json_output:
        typer.echo(json.dumps(values, ensure_ascii=False, sort_keys=True))
        return
    table = Table(title=project.name)
    table.add_column("Variant", style="bold cyan")
    table.add_column("Mode")
    table.add_column("Grid")
    table.add_column("Output")
    for item in project.variants:
        request = item.request
        dimensions = (
            f"{request.output_width or 'auto'}×{request.output_height or 'auto'}"
            if request.render_format.value in {"png", "svg"}
            else request.render_format.value
        )
        table.add_row(
            ("● " if item.identifier == project.active_variant else "  ")
            + f"{item.name} ({item.identifier})",
            request.mode,
            f"{request.width}×{request.height or 'auto'}",
            dimensions,
        )
    console.print(table)
    console.print(
        f"Source · {project.source.path} · "
        + ("available" if asset.is_file() else "[red]missing[/red]")
        + (
            " · [yellow]recovery available[/yellow]"
            if recovery_path(project_path).is_file()
            else ""
        )
    )


@project_app.command("render")
def render_project_command(
    project_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Atomic output path; previews glyph text when omitted.",
    ),
    variant: Optional[str] = typer.Option(
        None, "--variant", help="Variant identifier."
    ),
    recover: bool = typer.Option(
        True,
        "--recover/--saved-only",
        help="Render a compatible autosave when one exists.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit render metrics as JSON."
    ),
) -> None:
    """Render a saved or recovered project variant through the canonical engine."""

    try:
        session = ProjectSession.open(
            project_path, recover=recover, autosave_delay=None
        )
        selected = _variant(session.project, variant)
        artifact = render_image(
            session.project.source.resolve(project_path),
            selected.request,
            destination=output,
        )
        _remember_project(project_path)
        session.close(checkpoint=False)
    except (ProjectError, GlyphForgeRenderError) as exc:
        error_console.print(f"[bold red]Project render failed:[/bold red] {exc}")
        raise typer.Exit(2) from exc
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "project": str(project_path),
                    "variant": selected.identifier,
                    "output": str(output) if output is not None else None,
                    "metrics": artifact.metrics.to_dict(),
                },
                sort_keys=True,
            )
        )
    elif output is None:
        typer.echo(artifact.glyph_text)
    else:
        error_console.print(
            f"[green]Saved[/green] {output} · {artifact.columns}×{artifact.rows} cells"
        )


@project_app.command("variant-add")
def add_variant_command(
    project_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, resolve_path=True
    ),
    identifier: str = typer.Argument(..., help="Portable unique identifier."),
    name: Optional[str] = typer.Option(None, "--name", help="Friendly display name."),
    preset: Optional[Path] = typer.Option(
        None,
        "--preset",
        exists=True,
        dir_okay=False,
        resolve_path=True,
        help="Settings to use instead of copying the active variant.",
    ),
    activate: bool = typer.Option(True, "--activate/--keep-active"),
) -> None:
    """Add a non-destructive render variant."""

    try:
        with ProjectSession.open(project_path, autosave_delay=None) as session:
            request = load_preset(preset).request if preset is not None else None
            session.add_variant(
                identifier, name or identifier, request, activate=activate
            )
            session.save()
        _remember_project(project_path)
    except ProjectError as exc:
        raise typer.BadParameter(str(exc), param_hint="identifier") from exc
    error_console.print(f"[green]Added variant[/green] {identifier}")


@project_app.command("variant-select")
def select_variant_command(
    project_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, resolve_path=True
    ),
    identifier: str = typer.Argument(...),
) -> None:
    """Select a project's active render variant."""

    try:
        with ProjectSession.open(project_path, autosave_delay=None) as session:
            session.select_variant(identifier)
            session.save()
    except ProjectError as exc:
        raise typer.BadParameter(str(exc), param_hint="identifier") from exc
    error_console.print(f"[green]Selected[/green] {identifier}")


@project_app.command("variant-remove")
def remove_variant_command(
    project_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, resolve_path=True
    ),
    identifier: str = typer.Argument(...),
) -> None:
    """Remove a render variant while always retaining one."""

    try:
        with ProjectSession.open(project_path, autosave_delay=None) as session:
            session.remove_variant(identifier)
            session.save()
    except ProjectError as exc:
        raise typer.BadParameter(str(exc), param_hint="identifier") from exc
    error_console.print(f"[green]Removed[/green] {identifier}")


@project_app.command("recover")
def recover_project_command(
    project_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, resolve_path=True
    ),
    discard: bool = typer.Option(
        False,
        "--discard",
        help="Discard the autosave instead of promoting it.",
    ),
) -> None:
    """Promote a compatible autosave, or explicitly discard it."""

    try:
        if discard:
            session = ProjectSession.open(
                project_path, recover=False, autosave_delay=None
            )
            session.discard_recovery()
            session.close(checkpoint=False)
            error_console.print("[green]Discarded project recovery[/green]")
            return
        if not recovery_path(project_path).is_file():
            raise typer.BadParameter(
                "No project recovery is available", param_hint="project_path"
            )
        session = ProjectSession.open(project_path, recover=True, autosave_delay=None)
        session.save()
        session.close(checkpoint=False)
    except ProjectError as exc:
        error_console.print(f"[bold red]Recovery failed:[/bold red] {exc}")
        raise typer.Exit(2) from exc
    error_console.print("[green]Recovered and saved project[/green]")


@project_app.command("recent")
def recent_projects_command(
    prune: bool = typer.Option(
        False, "--prune", help="Remove paths that no longer exist."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON."
    ),
) -> None:
    """List bounded platform-native recent projects."""

    try:
        store = RecentProjectStore()
        if prune:
            store.prune()
        projects = store.list()
    except ProjectError as exc:
        error_console.print(
            f"[bold red]Could not read recent projects:[/bold red] {exc}"
        )
        raise typer.Exit(2) from exc
    if json_output:
        typer.echo(
            json.dumps(
                [
                    {"path": str(item.path), "accessed_at": item.accessed_at}
                    for item in projects
                ],
                sort_keys=True,
            )
        )
        return
    if not projects:
        console.print("[dim]No recent projects yet.[/dim]")
        return
    for item in projects:
        console.print(f"{item.accessed_at}  {item.path}")


@preset_app.command("create")
def create_preset_command(
    name: str = typer.Argument(..., help="Friendly preset name."),
    output: Path = typer.Argument(..., dir_okay=False, resolve_path=True),
    width: int = typer.Option(100, "--width", min=1, max=4096),
    height: Optional[int] = typer.Option(None, "--height", min=1, max=4096),
    mode: str = typer.Option("glyph", "--mode"),
    output_format: str = typer.Option("text", "--format"),
    output_width: Optional[int] = typer.Option(
        None, "--output-width", min=1, max=32768
    ),
    output_height: Optional[int] = typer.Option(
        None, "--output-height", min=1, max=32768
    ),
    fit: str = typer.Option("contain", "--fit-mode"),
    alignment: str = typer.Option("center", "--align"),
    charset: str = typer.Option("general", "--charset"),
    brightness: float = typer.Option(1.12, "--brightness", min=0, max=2),
    contrast: float = typer.Option(1.08, "--contrast", min=0, max=2),
) -> None:
    """Create a portable preset from explicit canonical render settings."""

    if output.exists():
        raise typer.BadParameter("Preset already exists", param_hint="output")
    try:
        preset = RenderPreset(
            name,
            RenderRequest(
                width=width,
                height=height,
                mode=mode,
                output_format=output_format,
                output_width=output_width,
                output_height=output_height,
                fit=fit,
                alignment=alignment,
                charset=charset,
                brightness=brightness,
                contrast=contrast,
            ),
        )
        save_preset(preset, output)
    except (ProjectError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="preset options") from exc
    error_console.print(f"[green]Saved preset[/green] {output}")


@preset_app.command("export")
def export_preset_command(
    project_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, resolve_path=True
    ),
    output: Path = typer.Argument(..., dir_okay=False, resolve_path=True),
    variant: Optional[str] = typer.Option(None, "--variant"),
    name: Optional[str] = typer.Option(None, "--name"),
) -> None:
    """Export one project variant as a reusable preset."""

    try:
        project = load_project(project_path)
        selected = _variant(project, variant)
        save_preset(
            RenderPreset(name or selected.name, selected.request),
            output,
        )
    except ProjectError as exc:
        raise typer.BadParameter(str(exc), param_hint="project_path") from exc
    error_console.print(f"[green]Exported preset[/green] {output}")


@preset_app.command("apply")
def apply_preset_command(
    preset_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, resolve_path=True
    ),
    project_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, resolve_path=True
    ),
    variant: Optional[str] = typer.Option(
        None,
        "--new-variant",
        help="Add these settings as a new variant instead of replacing the active one.",
    ),
) -> None:
    """Apply a preset to the active variant or add it non-destructively."""

    try:
        preset = load_preset(preset_path)
        with ProjectSession.open(project_path, autosave_delay=None) as session:
            if variant is None:
                session.update_active_request(preset.request)
            else:
                session.add_variant(variant, preset.name, preset.request)
            session.save()
    except ProjectError as exc:
        raise typer.BadParameter(str(exc), param_hint="preset_path") from exc
    error_console.print(f"[green]Applied preset[/green] {preset.name}")


@preset_app.command("info")
def preset_info_command(
    preset_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, resolve_path=True
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Inspect a portable render preset."""

    try:
        preset = load_preset(preset_path)
    except ProjectError as exc:
        raise typer.BadParameter(str(exc), param_hint="preset_path") from exc
    if json_output:
        typer.echo(json.dumps(preset.to_dict(), ensure_ascii=False, sort_keys=True))
        return
    request = preset.request
    console.print(
        f"[bold cyan]{preset.name}[/bold cyan] · {request.mode} · "
        f"{request.width}×{request.height or 'auto'} cells · "
        f"{request.render_format.value}"
    )


@preset_app.command("render")
def render_preset_command(
    preset_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, resolve_path=True
    ),
    source: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, resolve_path=True
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", dir_okay=False, resolve_path=True
    ),
) -> None:
    """Render one image using exactly the settings in a preset."""

    try:
        preset = load_preset(preset_path)
        artifact = render_image(source, preset.request, destination=output)
    except (ProjectError, GlyphForgeRenderError) as exc:
        error_console.print(f"[bold red]Preset render failed:[/bold red] {exc}")
        raise typer.Exit(2) from exc
    if output is None:
        typer.echo(artifact.glyph_text)
    else:
        error_console.print(f"[green]Saved[/green] {output}")


def batch_command(
    preset_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help=f"Shared render settings ({PRESET_SUFFIX}).",
    ),
    sources: list[Path] = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="One or more source images.",
    ),
    output_directory: Path = typer.Option(
        Path("glyph-forge-output"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    workers: int = typer.Option(1, "--workers", "-j", min=1, max=64),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Render a bounded, ordered image queue with one portable preset."""

    try:
        preset = load_preset(preset_path)
        items = items_for_sources(sources, output_directory, preset.request)

        def progress(update: BatchProgress) -> None:
            if not json_output:
                error_console.print(
                    f"  {update.completed}/{update.total} · "
                    f"{update.succeeded} saved · {update.failed} failed"
                )

        report = render_batch(
            items,
            workers=workers,
            progress=progress,
            fail_fast=fail_fast,
        )
    except (ProjectError, BatchError) as exc:
        error_console.print(f"[bold red]Batch failed:[/bold red] {exc}")
        raise typer.Exit(2) from exc
    if json_output:
        typer.echo(json.dumps(report.to_dict(), sort_keys=True))
    else:
        error_console.print(
            f"[green]Batch complete[/green] · {report.succeeded} saved · "
            f"{report.failed} failed · {report.elapsed:.2f}s"
        )
    if report.failed:
        raise typer.Exit(1)


__all__ = ["batch_command", "preset_app", "project_app"]

"""User-facing discovery and diagnostics for third-party extensions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from ..plugins.contracts import PLUGIN_API_VERSION, PLUGIN_ENTRY_POINT_GROUP

if TYPE_CHECKING:
    from ..plugins import PluginInfo

console = Console()
error_console = Console(stderr=True)

app = typer.Typer(
    name="plugins",
    help="Discover and diagnose optional third-party extensions.",
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _component_summary(info: PluginInfo) -> str:
    parts = []
    for label, values in (
        ("src", info.sources),
        ("render", info.renderers),
        ("transform", info.transforms),
        ("export", info.exporters),
    ):
        if values:
            parts.append(f"{label}:{','.join(values)}")
    return " · ".join(parts) or "—"


def _show_inventory(*, probe: bool, json_output: bool) -> None:
    from ..plugins import get_plugin_registry, plugins_enabled

    enabled = plugins_enabled()
    inventory = get_plugin_registry().inventory(load=probe)
    payload = {
        "api_version": PLUGIN_API_VERSION,
        "entry_point_group": PLUGIN_ENTRY_POINT_GROUP,
        "enabled": enabled,
        "plugins": [item.to_dict() for item in inventory],
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    if not enabled:
        console.print(
            "[yellow]External plugin discovery is disabled by "
            "GLYPH_FORGE_DISABLE_PLUGINS.[/yellow]"
        )
    table = Table(title=f"Glyph Forge plugins · API {PLUGIN_API_VERSION}")
    table.add_column("State")
    table.add_column("Identifier", style="bold cyan")
    table.add_column("Version")
    table.add_column("Components")
    table.add_column("Detail")
    for info in inventory:
        state_style = {
            "ready": "green",
            "discovered": "cyan",
            "error": "red",
            "conflict": "red",
        }.get(info.state, "yellow")
        detail = info.error or info.description or info.entry_point or ""
        table.add_row(
            f"[{state_style}]{info.state}[/{state_style}]",
            info.identifier,
            info.version or "—",
            _component_summary(info),
            detail,
        )
    if inventory:
        console.print(table)
    else:
        console.print(
            "No third-party plugins installed. Packages register the "
            f"[cyan]{PLUGIN_ENTRY_POINT_GROUP}[/cyan] entry-point group."
        )
    if not probe and inventory:
        console.print(
            "[dim]Metadata only; use --probe to import each plugin and validate "
            "its contract.[/dim]"
        )


@app.callback(invoke_without_command=True)
def plugins_callback(
    ctx: typer.Context,
    probe: bool = typer.Option(
        False,
        "--probe",
        help="Import every discovered plugin and isolate validation failures.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """List plugins without importing them unless ``--probe`` is supplied."""

    if ctx.invoked_subcommand is None:
        _show_inventory(probe=probe, json_output=json_output)


@app.command("list")
def list_plugins(
    probe: bool = typer.Option(
        False,
        "--probe",
        help="Import every discovered plugin and isolate validation failures.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """List installed plugin metadata and contribution names."""

    _show_inventory(probe=probe, json_output=json_output)


@app.command("inspect")
def inspect_plugin(
    identifier: str = typer.Argument(..., help="Installed entry-point identifier."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Load one plugin and show its validated contract."""

    from ..plugins import PluginError, get_plugin_registry

    try:
        info = get_plugin_registry().info(identifier, load=True)
    except PluginError as exc:
        if json_output:
            typer.echo(json.dumps({"identifier": identifier, "error": str(exc)}))
        else:
            error_console.print(f"[bold red]Plugin error:[/bold red] {exc}")
        raise typer.Exit(2) from exc

    if json_output:
        typer.echo(json.dumps(info.to_dict(), indent=2, sort_keys=True))
        return
    table = Table(show_header=False, box=None)
    table.add_column("Property", style="bold cyan")
    table.add_column("Value")
    table.add_row("Identifier", info.identifier)
    table.add_row("Name", info.name)
    table.add_row("Version", info.version)
    table.add_row("API", str(info.api_version))
    table.add_row("Distribution", info.distribution or "in-process")
    table.add_row("Sources", ", ".join(info.sources) or "—")
    table.add_row("Renderers", ", ".join(info.renderers) or "—")
    table.add_row("Transforms", ", ".join(info.transforms) or "—")
    table.add_row("Exporters", ", ".join(info.exporters) or "—")
    if info.description:
        table.add_row("Description", info.description)
    console.print(table)


__all__ = ["app", "inspect_plugin", "list_plugins", "plugins_callback"]

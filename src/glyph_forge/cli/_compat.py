"""Shared console-script forwarding with native per-command help."""

from __future__ import annotations

from collections.abc import Sequence

from typer.main import get_command


def run_unified_command(
    command_name: str,
    arguments: Sequence[str],
    *,
    program_name: str,
) -> int:
    """Invoke one unified subcommand as a standalone compatibility program."""

    from . import app

    root = get_command(app)
    commands = getattr(root, "commands", None)
    if not isinstance(commands, dict):  # pragma: no cover - Typer invariant
        raise RuntimeError("Glyph Forge's unified CLI did not create a command group")
    command = commands[command_name]
    try:
        command.main(
            args=list(arguments),
            prog_name=program_name,
            standalone_mode=True,
        )
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


__all__ = ["run_unified_command"]

"""CLI prompting utilities."""

from __future__ import annotations

from typing import Optional

import typer


def confirm_action(message: str, assume_yes: Optional[bool] = None, default: bool = False) -> bool:
    """Prompt for confirmation unless assume_yes is explicitly set."""
    if assume_yes is True:
        return True
    if assume_yes is False:
        return False
    return typer.confirm(message, default=default)

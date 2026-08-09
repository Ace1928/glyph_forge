"""Optional X11 virtual-display lifecycle for isolated GUI rendering."""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any


class VirtualDisplayError(RuntimeError):
    """Raised when an isolated display or its application cannot start."""


class VirtualDisplaySession:
    """Own one PyVirtualDisplay instance and restore the environment on exit."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        *,
        color_depth: int = 24,
        backend: str = "xvfb",
    ) -> None:
        if width < 1 or height < 1:
            raise ValueError("Virtual display dimensions must be positive")
        if color_depth not in {8, 16, 24, 32}:
            raise ValueError("Virtual display color depth must be 8, 16, 24, or 32")
        if backend not in {"xvfb", "xephyr", "xvnc"}:
            raise ValueError("Virtual display backend must be xvfb, xephyr, or xvnc")
        self.width = width
        self.height = height
        self.color_depth = color_depth
        self.backend = backend
        self._display: Any | None = None

    @property
    def active(self) -> bool:
        return self._display is not None

    @property
    def name(self) -> str:
        if self._display is None:
            return "virtual:inactive"
        return str(getattr(self._display, "new_display_var", "virtual:active"))

    def start(self) -> "VirtualDisplaySession":
        if self._display is not None:
            return self
        if platform.system() not in {"Linux", "FreeBSD"}:
            raise VirtualDisplayError(
                "Isolated virtual displays require an X11 host; use live screen "
                "or browser Studio on this operating system"
            )
        try:
            module = importlib.import_module("pyvirtualdisplay")
        except (ImportError, OSError) as exc:
            raise VirtualDisplayError(
                "Virtual application displays require PyVirtualDisplay and Xvfb; "
                "install glyph-forge[virtual] and your OS Xvfb package"
            ) from exc
        try:
            display = module.Display(
                backend=self.backend,
                visible=False,
                size=(self.width, self.height),
                color_depth=self.color_depth,
            )
            display.start()
        except Exception as exc:
            raise VirtualDisplayError(
                f"Could not start the {self.backend} virtual display: {exc}"
            ) from exc
        self._display = display
        return self

    def environment(self) -> dict[str, str]:
        """Return a child-process environment targeting this display."""

        if self._display is None:
            raise VirtualDisplayError("The virtual display is not active")
        env_method = getattr(self._display, "env", None)
        if callable(env_method):
            return dict(env_method())
        environment = os.environ.copy()
        environment["DISPLAY"] = str(self._display.new_display_var)
        return environment

    def launch(self, command: Sequence[str]) -> subprocess.Popen[bytes]:
        """Launch an argument-safe child process inside the active display."""

        arguments = [str(item) for item in command]
        if not arguments:
            raise ValueError("An application command is required")
        try:
            return subprocess.Popen(arguments, env=self.environment())
        except OSError as exc:
            raise VirtualDisplayError(
                f"Could not launch {arguments[0]!r}: {exc}"
            ) from exc

    def close(self) -> None:
        if self._display is None:
            return
        display, self._display = self._display, None
        try:
            display.stop()
        except Exception as exc:
            raise VirtualDisplayError(
                f"Could not stop the virtual display: {exc}"
            ) from exc

    def __enter__(self) -> "VirtualDisplaySession":
        return self.start()

    def __exit__(self, *_args: object) -> None:
        self.close()


@contextmanager
def virtual_display(
    width: int = 1280,
    height: int = 720,
    **options: Any,
) -> Iterator[VirtualDisplaySession]:
    """Provide an active isolated display for one lexical scope."""

    with VirtualDisplaySession(width, height, **options) as session:
        yield session


__all__ = ["VirtualDisplayError", "VirtualDisplaySession", "virtual_display"]

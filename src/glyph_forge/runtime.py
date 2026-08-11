"""Portable runtime discovery and adaptive performance defaults.

This module intentionally depends only on the Python standard library.  It can
therefore power ``glyph-forge doctor`` and choose safe defaults even when an
optional interface or media backend is not installed.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from importlib import metadata
from typing import Any, Iterable

PYTHON_DISTRIBUTION = "glyphforge"
STABLE_RELEASE_VERSION = "0.3.1"
STABLE_SOURCE_URL = (
    "https://github.com/Ace1928/glyph_forge/archive/refs/tags/"
    f"v{STABLE_RELEASE_VERSION}.zip"
)


def python_install_hint(extra: str | None = None) -> str:
    """Return the canonical install command for the current stable release."""

    distribution = f"{PYTHON_DISTRIBUTION}[{extra}]" if extra else PYTHON_DISTRIBUTION
    return f'pip install "{distribution} @ {STABLE_SOURCE_URL}"'


class PerformanceTier(str, Enum):
    """Named performance envelopes used throughout Glyph Forge."""

    ECO = "eco"
    BALANCED = "balanced"
    WORKSTATION = "workstation"


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Hardware-aware defaults that remain predictable and overridable."""

    tier: PerformanceTier
    cpu_count: int
    memory_bytes: int | None
    workers: int
    image_width: int
    stream_width: int
    target_fps: int
    resample: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        result = asdict(self)
        result["tier"] = self.tier.value
        return result


def subprocess_environment() -> dict[str, str] | None:
    """Return a safe child-process environment when the host requires one.

    Termux binaries carry their own runtime search paths.  An inherited
    ``LD_LIBRARY_PATH`` from another toolchain can take precedence and make a
    healthy FFmpeg installation fail at load time.  Android is the only host
    where Glyph Forge removes that override; every other platform inherits the
    caller's environment unchanged. Frozen applications retain the loader path
    for their own process but still use this clean environment for external
    tools.
    """

    if sys.platform != "android":
        return None
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    return environment


def configure_utf8_stdio() -> None:
    """Keep redirected Windows glyph output lossless.

    Windows can expose a legacy code-page encoding when a console program's
    output is captured or redirected.  Braille, block, and international glyph
    modes cannot be represented by those encodings, so CLI entry points switch
    their existing text streams to UTF-8 before producing output.  Other
    platforms already use UTF-8 in supported environments and remain untouched.
    """

    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            # Embedded hosts may expose a stream that cannot be reconfigured;
            # individual commands can still use their host-provided encoding.
            continue


def reexec_clean_android_environment(
    arguments: Iterable[str] | None = None,
) -> bool:
    """Relaunch the CLI once before Android loads optional native libraries.

    Changing ``LD_LIBRARY_PATH`` after process startup is too late for native
    modules loaded with ``dlopen`` (for example OpenCV).  Console entry points
    therefore relaunch themselves with the same arguments and a clean linker
    environment.  The second process naturally skips this path because the
    override is no longer present.

    PyInstaller sets a private library path required by its frozen process, so
    frozen applications skip the relaunch after their bootloader has started.
    """

    if (
        sys.platform != "android"
        or getattr(sys, "frozen", False)
        or not os.environ.get("LD_LIBRARY_PATH")
    ):
        return False

    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    values = list(sys.argv[1:] if arguments is None else arguments)
    launcher = sys.argv[0]
    if os.path.isfile(launcher) and os.path.basename(launcher) != "__main__.py":
        command = [sys.executable, os.path.abspath(launcher), *values]
    else:
        command = [sys.executable, "-m", "glyph_forge", *values]
    os.execve(sys.executable, command, environment)
    return True  # pragma: no cover - os.execve never returns on success


@dataclass(frozen=True, slots=True)
class Capability:
    """Availability and installation guidance for one feature dependency."""

    key: str
    label: str
    available: bool
    kind: str
    purpose: str
    install_hint: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


_PYTHON_CAPABILITIES: tuple[tuple[str, str, str, str | None], ...] = (
    ("PIL", "Pillow", "image conversion", python_install_hint()),
    ("numpy", "NumPy", "accelerated pixel mapping", python_install_hint()),
    ("pyfiglet", "pyfiglet", "text banners", python_install_hint()),
    ("rich", "Rich", "styled CLI output", python_install_hint()),
    ("typer", "Typer", "command-line interface", python_install_hint()),
    ("textual", "Textual", "terminal UI", python_install_hint("tui")),
    ("cv2", "OpenCV", "video and webcam capture", python_install_hint("media")),
    ("mss", "MSS", "cross-platform screen capture", python_install_hint("media")),
    (
        "yt_dlp",
        "yt-dlp",
        "video-site URL resolution",
        python_install_hint("network"),
    ),
    (
        "pyvirtualdisplay",
        "PyVirtualDisplay",
        "isolated X11 application displays",
        f"{python_install_hint('virtual')} and install Xvfb",
    ),
    (
        "pynput",
        "pynput",
        "explicit keyboard and pointer forwarding",
        f"{python_install_hint('control')} and grant OS input permission",
    ),
)

_TOOL_CAPABILITIES: tuple[tuple[str, str, str, str | None], ...] = (
    (
        "ffmpeg",
        "FFmpeg",
        "video encoding and audio muxing",
        "Install FFmpeg with your OS package manager",
    ),
    (
        "ffprobe",
        "ffprobe",
        "video metadata inspection",
        "Install FFmpeg with your OS package manager",
    ),
)


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _physical_memory() -> int | None:
    """Best-effort physical-memory detection without a required dependency."""

    try:
        import psutil  # type: ignore[import-untyped]

        return int(psutil.virtual_memory().total)
    except (ImportError, AttributeError, OSError):
        pass

    if hasattr(os, "sysconf"):
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page_size > 0:
                return pages * page_size
        except (OSError, TypeError, ValueError):
            pass
    return None


def _normalize_preference(
    preference: str | PerformanceTier | None,
) -> PerformanceTier | None:
    if isinstance(preference, PerformanceTier):
        return preference
    value = (preference or os.environ.get("GLYPH_FORGE_PERFORMANCE", "auto")).lower()
    aliases = {
        "low": PerformanceTier.ECO,
        "modest": PerformanceTier.ECO,
        "fast": PerformanceTier.ECO,
        "normal": PerformanceTier.BALANCED,
        "quality": PerformanceTier.WORKSTATION,
        "high": PerformanceTier.WORKSTATION,
    }
    if value in aliases:
        return aliases[value]
    if value == "auto":
        return None
    try:
        return PerformanceTier(value)
    except ValueError as exc:
        choices = "auto, eco, balanced, workstation"
        raise ValueError(
            f"Unknown performance mode {value!r}; choose {choices}"
        ) from exc


def detect_runtime_profile(
    preference: str | PerformanceTier | None = None,
    *,
    cpu_count: int | None = None,
    memory_bytes: int | None = None,
) -> RuntimeProfile:
    """Choose conservative defaults for the detected machine.

    Explicit ``cpu_count`` and ``memory_bytes`` values exist primarily for
    deterministic callers and tests.  Every returned value is a default rather
    than a hard limit; individual commands may override it.
    """

    cpus = max(1, int(cpu_count if cpu_count is not None else (os.cpu_count() or 1)))
    memory = memory_bytes if memory_bytes is not None else _physical_memory()
    requested = _normalize_preference(preference)

    if requested is None:
        gib = memory / (1024**3) if memory is not None else None
        if cpus <= 2 or (gib is not None and gib < 3):
            tier = PerformanceTier.ECO
        elif cpus >= 12 and (gib is None or gib >= 12):
            tier = PerformanceTier.WORKSTATION
        else:
            tier = PerformanceTier.BALANCED
    else:
        tier = requested

    if tier is PerformanceTier.ECO:
        return RuntimeProfile(tier, cpus, memory, 1, 72, 64, 12, "bilinear")
    if tier is PerformanceTier.WORKSTATION:
        return RuntimeProfile(
            tier,
            cpus,
            memory,
            min(16, cpus),
            160,
            160,
            30,
            "lanczos",
        )
    return RuntimeProfile(
        tier,
        cpus,
        memory,
        min(4, cpus),
        100,
        100,
        20,
        "bicubic",
    )


def iter_capabilities() -> Iterable[Capability]:
    """Yield all optional and required runtime capability checks."""

    for module, label, purpose, hint in _PYTHON_CAPABILITIES:
        yield Capability(
            key=module,
            label=label,
            available=_module_available(module),
            kind="python",
            purpose=purpose,
            install_hint=hint,
        )
    for command, label, purpose, hint in _TOOL_CAPABILITIES:
        available, detail = _probe_tool(command)
        yield Capability(
            key=command,
            label=label,
            available=available,
            kind="tool",
            purpose=purpose,
            install_hint=hint,
            detail=detail,
        )


def _probe_tool(command: str) -> tuple[bool, str | None]:
    """Confirm a media executable can start, not merely that a path exists."""

    executable = shutil.which(command)
    if executable is None:
        return False, "not found on PATH"
    try:
        result = subprocess.run(
            [executable, "-version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=3,
            text=True,
            env=subprocess_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"found at {executable}, but self-check failed: {exc}"
    if result.returncode:
        error = (result.stderr or "").strip().splitlines()
        reason = error[-1] if error else f"exit status {result.returncode}"
        return False, f"found at {executable}, but cannot run: {reason}"
    return True, executable


def package_version() -> str:
    """Return the installed distribution version without importing the package."""

    try:
        return metadata.version("glyphforge")
    except metadata.PackageNotFoundError:
        return "0.3.1.dev0"


def runtime_report(preference: str | PerformanceTier | None = None) -> dict[str, Any]:
    """Collect a stable, JSON-ready diagnostics report."""

    profile = detect_runtime_profile(preference)
    return {
        "glyph_forge": package_version(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system() or os.name,
        "machine": platform.machine() or "unknown",
        "profile": profile.to_dict(),
        "capabilities": [capability.to_dict() for capability in iter_capabilities()],
    }


__all__ = [
    "Capability",
    "PerformanceTier",
    "PYTHON_DISTRIBUTION",
    "RuntimeProfile",
    "STABLE_RELEASE_VERSION",
    "STABLE_SOURCE_URL",
    "configure_utf8_stdio",
    "detect_runtime_profile",
    "iter_capabilities",
    "package_version",
    "python_install_hint",
    "reexec_clean_android_environment",
    "runtime_report",
    "subprocess_environment",
]

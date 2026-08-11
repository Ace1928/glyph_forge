# -*- mode: python ; coding: utf-8 -*-
"""One-directory Glyph Forge bundle for fast startup and inspectable contents."""

import os
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

tool_root = Path(SPECPATH)
project_root = tool_root.parent

datas = (
    collect_data_files("glyph_forge")
    + collect_data_files("pyfiglet")
    + copy_metadata("glyphforge")
)
hidden_imports = [*collect_submodules("glyph_forge"), "pyfiglet.fonts"]
core_excludes = (
    ["cv2", "mss", "pynput", "pyvirtualdisplay", "yt_dlp"]
    if os.environ.get("GLYPH_FORGE_BUNDLE_CORE") == "1"
    else []
)

analysis = Analysis(
    [str(tool_root / "bundle_launcher.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["mypy", "pytest", "ruff", "twine", *core_excludes],
    noarchive=False,
    optimize=1,
)
python_archive = PYZ(analysis.pure)

executable = EXE(
    python_archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="glyph-forge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="glyph-forge",
)

"""Tests for portable runtime discovery and the unified CLI foundation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from glyph_forge import runtime
from glyph_forge.cli import app
from glyph_forge.runtime import PerformanceTier, detect_runtime_profile, runtime_report

runner = CliRunner()


def test_runtime_profile_scales_from_modest_to_workstation() -> None:
    modest = detect_runtime_profile(cpu_count=2, memory_bytes=2 * 1024**3)
    workstation = detect_runtime_profile(cpu_count=24, memory_bytes=32 * 1024**3)

    assert modest.tier is PerformanceTier.ECO
    assert modest.workers == 1
    assert workstation.tier is PerformanceTier.WORKSTATION
    assert workstation.workers <= workstation.cpu_count
    assert workstation.stream_width > modest.stream_width
    assert workstation.target_fps > modest.target_fps


def test_runtime_profile_honors_explicit_mode() -> None:
    profile = detect_runtime_profile(
        "balanced", cpu_count=64, memory_bytes=128 * 1024**3
    )

    assert profile.tier is PerformanceTier.BALANCED
    assert profile.workers == 4


def test_runtime_report_is_json_serializable() -> None:
    report = runtime_report("eco")

    assert report["profile"]["tier"] == "eco"
    assert any(item["key"] == "PIL" for item in report["capabilities"])
    json.dumps(report)


def test_tool_probe_rejects_a_binary_that_exists_but_cannot_start(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runtime.shutil, "which", lambda _command: "/tools/ffmpeg")
    completed = subprocess.CompletedProcess(
        ["/tools/ffmpeg", "-version"],
        returncode=127,
        stderr="shared library could not be loaded\n",
    )
    monkeypatch.setattr(runtime.subprocess, "run", lambda *_args, **_kwargs: completed)

    available, detail = runtime._probe_tool("ffmpeg")

    assert not available
    assert detail is not None and "cannot run" in detail


def test_android_subprocess_environment_removes_foreign_library_path(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runtime.sys, "platform", "android")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/foreign/toolchain")
    monkeypatch.setenv("GLYPH_FORGE_TEST", "preserved")

    environment = runtime.subprocess_environment()

    assert environment is not None
    assert "LD_LIBRARY_PATH" not in environment
    assert environment["GLYPH_FORGE_TEST"] == "preserved"


def test_non_android_subprocess_environment_inherits_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(runtime.sys, "platform", "linux")

    assert runtime.subprocess_environment() is None


def test_package_import_is_lazy_and_has_no_home_side_effects(tmp_path: Path) -> None:
    source_root = Path(__file__).parents[1] / "src"
    environment = os.environ.copy()
    environment["HOME"] = str(tmp_path)
    environment["PYTHONPATH"] = str(source_root)
    command = [
        sys.executable,
        "-c",
        (
            "import json, sys, glyph_forge; "
            "print(json.dumps({'pil': 'PIL' in sys.modules, "
            "'numpy': 'numpy' in sys.modules, 'version': glyph_forge.__version__}))"
        ),
    ]

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    result = json.loads(completed.stdout)

    assert result == {"pil": False, "numpy": False, "version": "0.2.0"}
    assert not (tmp_path / ".glyph_forge").exists()


def test_cli_home_and_version_json() -> None:
    home = runner.invoke(app, [])
    version = runner.invoke(app, ["version", "--json"])

    assert home.exit_code == 0
    assert "Quick start" in home.stdout
    assert version.exit_code == 0
    assert json.loads(version.stdout)["glyph_forge"] == "0.2.0"


def test_cli_doctor_json() -> None:
    result = runner.invoke(app, ["doctor", "--performance", "eco", "--json"])

    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["profile"]["tier"] == "eco"
    assert isinstance(report["capabilities"], list)


def test_cli_image_command_saves_output(tmp_path: Path) -> None:
    source = tmp_path / "gradient.png"
    destination = tmp_path / "gradient.txt"
    Image.linear_gradient("L").resize((32, 16)).save(source)

    result = runner.invoke(
        app,
        [
            "image",
            str(source),
            "--output",
            str(destination),
            "--width",
            "16",
            "--height",
            "8",
            "--no-fit",
            "--no-preview",
        ],
    )

    assert result.exit_code == 0
    assert destination.exists()
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8
    assert all(len(line) == 16 for line in lines)


def test_cli_rejects_unknown_performance_mode(tmp_path: Path) -> None:
    source = tmp_path / "sample.png"
    Image.new("L", (4, 4), 128).save(source)

    result = runner.invoke(app, ["image", str(source), "--performance", "warp"])

    assert result.exit_code != 0
    assert "Unknown performance mode" in result.output

"""Tests for portable runtime discovery and the unified CLI foundation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

import glyph_forge
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


def test_package_version_uses_the_unambiguous_distribution_name(monkeypatch) -> None:
    requested: list[str] = []

    def fake_version(distribution: str) -> str:
        requested.append(distribution)
        return "9.8.7"

    monkeypatch.setattr(runtime.metadata, "version", fake_version)

    assert runtime.package_version() == "9.8.7"
    assert requested == ["glyphforge"]


def test_install_hint_tracks_the_stable_product_version() -> None:
    command = runtime.python_install_hint("media")

    assert runtime.STABLE_RELEASE_VERSION == glyph_forge.__version__
    assert "glyphforge[media]" in command
    assert f"/tags/v{glyph_forge.__version__}.zip" in command
    assert "glyph-forge[" not in command


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


def test_windows_cli_streams_are_reconfigured_for_lossless_glyphs(
    monkeypatch,
) -> None:
    calls: list[dict[str, str]] = []

    class Stream:
        def reconfigure(self, **options: str) -> None:
            calls.append(options)

    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setattr(runtime.sys, "stdout", Stream())
    monkeypatch.setattr(runtime.sys, "stderr", Stream())

    runtime.configure_utf8_stdio()

    assert calls == [{"encoding": "utf-8"}, {"encoding": "utf-8"}]


def test_non_windows_cli_streams_keep_host_configuration(monkeypatch) -> None:
    class Stream:
        def reconfigure(self, **_options: str) -> None:
            pytest.fail("non-Windows streams must remain untouched")

    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime.sys, "stdout", Stream())
    monkeypatch.setattr(runtime.sys, "stderr", Stream())

    runtime.configure_utf8_stdio()


def test_frozen_android_app_keeps_its_bootloader_library_path(monkeypatch) -> None:
    monkeypatch.setattr(runtime.sys, "platform", "android")
    monkeypatch.setattr(runtime.sys, "frozen", True, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/bundle/_internal")
    monkeypatch.setattr(
        runtime.os,
        "execve",
        lambda *_args: pytest.fail("a frozen app must not relaunch itself"),
    )

    assert runtime.reexec_clean_android_environment() is False


@pytest.mark.parametrize("module_entry", [False, True])
def test_android_cli_reexecs_before_loading_native_modules(
    tmp_path: Path,
    monkeypatch,
    module_entry: bool,
) -> None:
    class Relaunched(RuntimeError):
        pass

    launcher = tmp_path / ("__main__.py" if module_entry else "glyph-forge")
    launcher.write_text("# launcher\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_execve(
        executable: str,
        arguments: list[str],
        environment: dict[str, str],
    ) -> None:
        captured.update(
            executable=executable,
            arguments=arguments,
            environment=environment,
        )
        raise Relaunched

    monkeypatch.setattr(runtime.sys, "platform", "android")
    monkeypatch.setattr(runtime.sys, "argv", [str(launcher), "video", "clip.mp4"])
    monkeypatch.setattr(runtime.os, "execve", fake_execve)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/foreign/toolchain")
    monkeypatch.setenv("GLYPH_FORGE_TEST", "preserved")

    with pytest.raises(Relaunched):
        runtime.reexec_clean_android_environment()

    expected_prefix = (
        [runtime.sys.executable, "-m", "glyph_forge"]
        if module_entry
        else [runtime.sys.executable, str(launcher)]
    )
    assert captured["executable"] == runtime.sys.executable
    assert captured["arguments"] == [*expected_prefix, "video", "clip.mp4"]
    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert "LD_LIBRARY_PATH" not in environment
    assert environment["GLYPH_FORGE_TEST"] == "preserved"


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

    assert result == {"pil": False, "numpy": False, "version": "0.4.0"}
    assert not (tmp_path / ".glyph_forge").exists()


def test_cli_home_and_version_json() -> None:
    home = runner.invoke(app, [])
    version = runner.invoke(app, ["version", "--json"])

    assert home.exit_code == 0
    assert "Quick start" in home.stdout
    assert version.exit_code == 0
    assert json.loads(version.stdout)["glyph_forge"] == "0.4.0"


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

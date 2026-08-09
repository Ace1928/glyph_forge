"""Tests for portable and deterministic release assets."""

from __future__ import annotations

import subprocess
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

import glyph_forge
from tools import release_assets


def make_bundle(root: Path) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    executable = bundle / "glyph-forge"
    executable.write_bytes(b"portable executable\n")
    resources = bundle / "_internal" / "glyph_forge" / "ui" / "web"
    resources.mkdir(parents=True)
    (resources / "index.html").write_text("<h1>Glyph Forge</h1>\n", encoding="utf-8")
    (bundle / "empty").mkdir()
    return bundle


def test_project_version_and_tag_are_canonical() -> None:
    assert release_assets.project_version() == glyph_forge.__version__
    assert release_assets.verify_tag(f"v{glyph_forge.__version__}") == (
        f"v{glyph_forge.__version__}"
    )
    with pytest.raises(release_assets.ReleaseError, match="must exactly match"):
        release_assets.verify_tag("v999.0.0")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Linux-X64", "linux-x86_64"),
        ("Windows-AMD64", "windows-x86_64"),
        ("macOS-ARM64", "macos-arm64"),
        ("Darwin-aarch64", "macos-arm64"),
    ],
)
def test_platform_names_are_stable(value: str, expected: str) -> None:
    assert release_assets.normalize_platform(value) == expected


@pytest.mark.parametrize("value", ["linux", "Plan9-X64", "Linux-mips"])
def test_unsupported_platform_names_are_rejected(value: str) -> None:
    with pytest.raises(release_assets.ReleaseError):
        release_assets.normalize_platform(value)


@pytest.mark.parametrize(
    ("platform", "suffix"),
    [("Windows-X64", ".zip"), ("Linux-X64", ".tar.gz")],
)
def test_portable_archives_are_byte_reproducible(
    tmp_path: Path, platform: str, suffix: str
) -> None:
    bundle = make_bundle(tmp_path)
    first = release_assets.archive_bundle(
        bundle, tmp_path / "first", platform, version="1.2.3", epoch=1_700_000_000
    )
    second = release_assets.archive_bundle(
        bundle, tmp_path / "second", platform, version="1.2.3", epoch=1_700_000_000
    )

    assert first.name.endswith(suffix)
    assert release_assets.sha256(first) == release_assets.sha256(second)
    if suffix == ".zip":
        with zipfile.ZipFile(first) as archive:
            names = archive.namelist()
            executable = archive.getinfo("glyph-forge-1.2.3/glyph-forge")
            assert executable.external_attr >> 16 & 0o777 == 0o755
    else:
        with tarfile.open(first) as archive:
            names = archive.getnames()
            executable = archive.getmember("glyph-forge-1.2.3/glyph-forge")
            assert executable.mode == 0o755
    assert "glyph-forge-1.2.3/_internal/glyph_forge/ui/web/index.html" in names
    assert "glyph-forge-1.2.3/empty" in names or ("glyph-forge-1.2.3/empty/" in names)


def test_archive_cannot_be_written_inside_bundle(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    with pytest.raises(release_assets.ReleaseError, match="inside"):
        release_assets.archive_bundle(bundle, bundle, "Linux-X64", version="1.0.0")


def test_archive_rejects_escaping_symlink(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("private", encoding="utf-8")
    try:
        (bundle / "escape").symlink_to(outside)
    except OSError:
        pytest.skip("This platform does not permit test symlinks")

    with pytest.raises(release_assets.ReleaseError, match="absolute symlink"):
        release_assets.archive_bundle(
            bundle, tmp_path / "output", "Linux-X64", version="1.0.0"
        )


def test_archive_preserves_safe_relative_symlink(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    try:
        (bundle / "glyph-forge-link").symlink_to("glyph-forge")
    except OSError:
        pytest.skip("This platform does not permit test symlinks")

    output = release_assets.archive_bundle(
        bundle, tmp_path / "output", "Linux-X64", version="1.0.0"
    )

    with tarfile.open(output) as archive:
        link = archive.getmember("glyph-forge-1.0.0/glyph-forge-link")
    assert link.issym()
    assert link.linkname == "glyph-forge"


def test_archive_rejects_relative_symlink_escape(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    (tmp_path / "outside").write_text("private", encoding="utf-8")
    try:
        (bundle / "escape").symlink_to("../outside")
    except OSError:
        pytest.skip("This platform does not permit test symlinks")

    with pytest.raises(release_assets.ReleaseError, match="escapes"):
        release_assets.archive_bundle(
            bundle, tmp_path / "output", "Linux-X64", version="1.0.0"
        )


def test_reproducible_directory_comparison_reports_differences(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "asset.whl").write_bytes(b"same")
    (second / "asset.whl").write_bytes(b"same")

    assert release_assets.compare_directories(first, second) == {
        "asset.whl": release_assets.sha256(first / "asset.whl")
    }
    (second / "asset.whl").write_bytes(b"different")
    with pytest.raises(release_assets.ReleaseError, match="asset.whl"):
        release_assets.compare_directories(first, second)


def write_test_sdist(path: Path, *, mtime: int, name: str = "project/file.txt") -> None:
    with tarfile.open(path, "w:gz") as archive:
        root = tarfile.TarInfo("project")
        root.type = tarfile.DIRTYPE
        root.mode = 0o775
        root.uid = 1000
        root.gid = 1000
        root.mtime = mtime
        archive.addfile(root)
        payload = b"release source\n"
        member = tarfile.TarInfo(name)
        member.size = len(payload)
        member.mode = 0o664
        member.uid = 1000
        member.gid = 1000
        member.mtime = mtime
        archive.addfile(member, BytesIO(payload))


def test_sdist_normalization_is_safe_and_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    write_test_sdist(first, mtime=100)
    write_test_sdist(second, mtime=200)

    release_assets.normalize_sdist(first, epoch=1_700_000_000)
    release_assets.normalize_sdist(second, epoch=1_700_000_000)

    assert release_assets.sha256(first) == release_assets.sha256(second)
    with tarfile.open(first) as archive:
        root = archive.getmember("project")
        member = archive.getmember("project/file.txt")
    assert (root.uid, root.gid, root.mode, root.mtime) == (
        0,
        0,
        0o755,
        1_700_000_000,
    )
    assert (member.uid, member.gid, member.mode, member.mtime) == (
        0,
        0,
        0o644,
        1_700_000_000,
    )


@pytest.mark.parametrize("name", ["../escape", r"..\escape", "/absolute"])
def test_sdist_normalization_rejects_traversal(tmp_path: Path, name: str) -> None:
    source = tmp_path / "unsafe.tar.gz"
    write_test_sdist(source, mtime=100, name=name)

    with pytest.raises(release_assets.ReleaseError, match="Unsafe"):
        release_assets.normalize_sdist(source, epoch=1_700_000_000)

    assert source.is_file()


def test_checksum_manifest_is_sorted_and_excludes_itself(tmp_path: Path) -> None:
    (tmp_path / "z.zip").write_bytes(b"z")
    (tmp_path / "a.whl").write_bytes(b"a")
    (tmp_path / "ignored.sha256").write_text("old", encoding="utf-8")
    output = tmp_path / "SHA256SUMS"

    checksums = release_assets.write_checksums(tmp_path, output)

    assert [name for _digest, name in checksums] == ["a.whl", "z.zip"]
    assert output.read_text(encoding="utf-8").splitlines() == [
        f"{release_assets.sha256(tmp_path / 'a.whl')}  a.whl",
        f"{release_assets.sha256(tmp_path / 'z.zip')}  z.zip",
    ]


def test_bundle_executable_supports_unix_and_windows_names(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    windows = bundle / "glyph-forge.exe"
    windows.write_bytes(b"exe")
    assert release_assets.bundle_executable(bundle) == windows
    windows.unlink()
    with pytest.raises(release_assets.ReleaseError, match="missing"):
        release_assets.bundle_executable(bundle)


def test_bundle_smoke_checks_version_demo_and_studio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "glyph-forge"
    executable.write_bytes(b"executable")
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        stdout = (
            f'{{"glyph_forge":"{glyph_forge.__version__}"}}\n'
            if command[1:] == ["version", "--json"]
            else "ok\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(release_assets, "bundle_executable", lambda _bundle: executable)
    monkeypatch.setattr(release_assets.subprocess, "run", fake_run)

    report = release_assets.smoke_bundle(tmp_path)

    assert len(report) == 3
    assert [command[1] for command in calls] == ["version", "demo", "studio"]
    assert calls[-1][-3:] == ["--no-open", "--duration", "0.05"]


def test_bundle_smoke_wraps_version_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        release_assets,
        "bundle_executable",
        lambda _bundle: tmp_path / "glyph-forge",
    )
    monkeypatch.setattr(
        release_assets.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("glyph-forge", 90)
        ),
    )

    with pytest.raises(release_assets.ReleaseError, match="version check"):
        release_assets.smoke_bundle(tmp_path)


def test_release_cli_returns_concise_errors(capsys: pytest.CaptureFixture[str]) -> None:
    result = release_assets.main(["verify-tag", "--tag", "wrong"])

    assert result == 2
    assert "release error:" in capsys.readouterr().err


def test_bundle_spec_and_release_workflow_cover_every_supported_os() -> None:
    spec = (release_assets.PROJECT_ROOT / "tools" / "glyph-forge.spec").read_text(
        encoding="utf-8"
    )
    compile(spec, "glyph-forge.spec", "exec")
    assert 'collect_data_files("pyfiglet")' in spec
    workflow = (
        release_assets.PROJECT_ROOT / ".github" / "workflows" / "release.yml"
    ).read_text(encoding="utf-8")

    for runner in ("ubuntu-latest", "windows-latest", "macos-latest"):
        assert runner in workflow
    assert '- "v*"' in workflow
    assert "verify-tag --tag" in workflow
    assert "actions/attest@v4" in workflow
    assert "SHA256SUMS" in workflow
    ci_workflow = (
        release_assets.PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    assert 'GLYPH_FORGE_BUNDLE_CORE: "1"' in ci_workflow

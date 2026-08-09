"""Deterministic, cross-platform Glyph Forge release helpers."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 job
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINIMUM_ZIP_EPOCH = 315532800  # 1980-01-01, the first timestamp ZIP supports.
COPY_CHUNK_BYTES = 1024 * 1024


class ReleaseError(RuntimeError):
    """Raised when release inputs or generated assets are unsafe or inconsistent."""


def project_version(pyproject: Path | None = None) -> str:
    """Read the canonical project version without importing the source package."""

    source = pyproject or PROJECT_ROOT / "pyproject.toml"
    with source.open("rb") as stream:
        data = tomllib.load(stream)
    try:
        version = str(data["project"]["version"])
    except (KeyError, TypeError) as exc:
        raise ReleaseError(f"Missing [project].version in {source}") from exc
    if not version or any(character.isspace() for character in version):
        raise ReleaseError(f"Invalid project version: {version!r}")
    return version


def verify_tag(tag: str, *, version: str | None = None) -> str:
    """Require an exact ``vVERSION`` tag before release publication."""

    expected = f"v{version or project_version()}"
    if tag != expected:
        raise ReleaseError(f"Release tag {tag!r} must exactly match {expected!r}")
    return expected


def normalize_platform(value: str) -> str:
    """Normalize GitHub runner OS/architecture labels for public filenames."""

    try:
        system, architecture = value.split("-", 1)
    except ValueError as exc:
        raise ReleaseError("Platform must use OS-ARCH form") from exc
    systems = {
        "darwin": "macos",
        "linux": "linux",
        "macos": "macos",
        "windows": "windows",
    }
    architectures = {
        "aarch64": "arm64",
        "amd64": "x86_64",
        "arm64": "arm64",
        "x64": "x86_64",
        "x86_64": "x86_64",
    }
    normalized_system = systems.get(system.casefold())
    normalized_architecture = architectures.get(architecture.casefold())
    if normalized_system is None or normalized_architecture is None:
        raise ReleaseError(f"Unsupported release platform: {value}")
    return f"{normalized_system}-{normalized_architecture}"


def _normalized_mode(path: Path) -> int:
    if path.is_dir():
        return 0o755
    return 0o755 if path.stat().st_mode & 0o111 else 0o644


def _safe_link_target(path: Path, source: Path) -> str:
    target = os.readlink(path)
    if Path(target).is_absolute():
        raise ReleaseError(f"Bundle contains an absolute symlink: {path}")
    resolved = (path.parent / target).resolve(strict=False)
    try:
        resolved.relative_to(source)
    except ValueError as exc:
        raise ReleaseError(
            f"Bundle symlink escapes its root: {path} -> {target}"
        ) from exc
    return target


def _bundle_members(source: Path) -> list[Path]:
    resolved = source.resolve(strict=True)
    if not resolved.is_dir():
        raise ReleaseError(f"Bundle source is not a directory: {source}")
    members = [resolved, *resolved.rglob("*")]
    return sorted(members, key=lambda item: item.relative_to(resolved).as_posix())


def _archive_name(path: Path, source: Path, root_name: str) -> str:
    if path == source:
        return root_name
    return f"{root_name}/{path.relative_to(source).as_posix()}"


def _copy_stream(source: BinaryIO, destination: BinaryIO) -> None:
    shutil.copyfileobj(source, destination, length=COPY_CHUNK_BYTES)


def _write_zip(source: Path, output: Path, *, epoch: int, root_name: str) -> None:
    zip_epoch = max(epoch, MINIMUM_ZIP_EPOCH)
    timestamp = time.gmtime(zip_epoch)[:6]
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in _bundle_members(source):
            name = _archive_name(path, source, root_name)
            if path.is_dir() and not path.is_symlink():
                name = f"{name}/"
            info = zipfile.ZipInfo(name, date_time=timestamp)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            if path.is_symlink():
                target = _safe_link_target(path, source)
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, target.encode("utf-8"))
            elif path.is_dir():
                info.external_attr = (stat.S_IFDIR | 0o755) << 16 | 0x10
                archive.writestr(info, b"")
            else:
                info.external_attr = (stat.S_IFREG | _normalized_mode(path)) << 16
                with (
                    path.open("rb") as input_stream,
                    archive.open(info, "w") as output_stream,
                ):
                    _copy_stream(input_stream, output_stream)


def _tar_info(path: Path, source: Path, root_name: str, epoch: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(_archive_name(path, source, root_name))
    info.mtime = epoch
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    if path.is_symlink():
        info.type = tarfile.SYMTYPE
        info.mode = 0o777
        info.linkname = _safe_link_target(path, source)
    elif path.is_dir():
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
    else:
        info.type = tarfile.REGTYPE
        info.mode = _normalized_mode(path)
        info.size = path.stat().st_size
    return info


def _write_tar(source: Path, output: Path, *, epoch: int, root_name: str) -> None:
    with (
        output.open("wb") as raw_stream,
        gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_stream,
            compresslevel=9,
            mtime=epoch,
        ) as compressed,
        tarfile.open(
            fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
        ) as archive,
    ):
        for path in _bundle_members(source):
            info = _tar_info(path, source, root_name, epoch)
            if info.isreg():
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
            else:
                archive.addfile(info)


def archive_bundle(
    source: Path,
    output_directory: Path,
    platform: str,
    *,
    version: str | None = None,
    epoch: int | None = None,
) -> Path:
    """Create a deterministic ZIP or tarball for one PyInstaller directory."""

    resolved_source = source.resolve(strict=True)
    normalized_platform = normalize_platform(platform)
    selected_version = version or project_version()
    selected_epoch = (
        epoch
        if epoch is not None
        else int(os.environ.get("SOURCE_DATE_EPOCH", "315532800"))
    )
    if selected_epoch < 0:
        raise ReleaseError("Archive epoch cannot be negative")
    output_directory.mkdir(parents=True, exist_ok=True)
    extension = ".zip" if normalized_platform.startswith("windows-") else ".tar.gz"
    output = output_directory / (
        f"glyph-forge-{selected_version}-{normalized_platform}{extension}"
    )
    try:
        output.resolve().relative_to(resolved_source)
    except ValueError:
        pass
    else:
        raise ReleaseError("Release archive cannot be written inside its bundle")
    root_name = f"glyph-forge-{selected_version}"
    if extension == ".zip":
        _write_zip(resolved_source, output, epoch=selected_epoch, root_name=root_name)
    else:
        _write_tar(resolved_source, output, epoch=selected_epoch, root_name=root_name)
    return output


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def compare_directories(first: Path, second: Path) -> dict[str, str]:
    """Require two build directories to contain byte-identical regular files."""

    def inventory(root: Path) -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): sha256(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    first_files = inventory(first)
    second_files = inventory(second)
    if first_files != second_files:
        names = sorted(set(first_files) | set(second_files))
        differences = [
            name for name in names if first_files.get(name) != second_files.get(name)
        ]
        raise ReleaseError(
            "Release builds are not reproducible: " + ", ".join(differences)
        )
    if not first_files:
        raise ReleaseError("Release build directory contains no files")
    return first_files


def _safe_archive_member(name: str) -> None:
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts:
        raise ReleaseError(f"Unsafe sdist archive member: {name!r}")


def normalize_sdist(source: Path, *, epoch: int | None = None) -> Path:
    """Repack a source distribution with deterministic, safe tar metadata."""

    selected_epoch = (
        epoch
        if epoch is not None
        else int(os.environ.get("SOURCE_DATE_EPOCH", "315532800"))
    )
    if selected_epoch < 0:
        raise ReleaseError("Archive epoch cannot be negative")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=source.parent, prefix=f".{source.name}.", suffix=".normalized"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with (
            tarfile.open(source, "r:gz") as input_archive,
            temporary.open("wb") as raw_stream,
            gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_stream,
                compresslevel=9,
                mtime=selected_epoch,
            ) as compressed,
            tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
            ) as output_archive,
        ):
            members = sorted(input_archive.getmembers(), key=lambda item: item.name)
            if not members:
                raise ReleaseError(f"Source distribution is empty: {source}")
            for member in members:
                _safe_archive_member(member.name)
                if not (member.isfile() or member.isdir()):
                    raise ReleaseError(
                        f"Unsupported sdist member type: {member.name!r}"
                    )
                normalized = copy.copy(member)
                normalized.uid = 0
                normalized.gid = 0
                normalized.uname = ""
                normalized.gname = ""
                normalized.mtime = selected_epoch
                normalized.pax_headers = {}
                normalized.mode = (
                    0o755 if member.isdir() or member.mode & 0o111 else 0o644
                )
                if member.isfile():
                    stream = input_archive.extractfile(member)
                    if stream is None:
                        raise ReleaseError(
                            f"Could not read sdist member: {member.name}"
                        )
                    with stream:
                        output_archive.addfile(normalized, stream)
                else:
                    output_archive.addfile(normalized)
        os.replace(temporary, source)
        source.chmod(0o644)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return source


def write_checksums(directory: Path, output: Path) -> list[tuple[str, str]]:
    """Write a stable SHA256SUMS file for every top-level release asset."""

    resolved_output = output.resolve()
    assets = [
        path
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file()
        and not path.is_symlink()
        and path.resolve() != resolved_output
        and not path.name.endswith(".sha256")
    ]
    if not assets:
        raise ReleaseError(f"No release assets found in {directory}")
    checksums = [(sha256(path), path.name) for path in assets]
    output.write_text(
        "".join(f"{digest}  {name}\n" for digest, name in checksums),
        encoding="utf-8",
        newline="\n",
    )
    return checksums


def bundle_executable(bundle: Path) -> Path:
    """Locate the platform-specific executable in a one-directory bundle."""

    candidates = [bundle / "glyph-forge", bundle / "glyph-forge.exe"]
    executable = next(
        (candidate for candidate in candidates if candidate.is_file()), None
    )
    if executable is None:
        raise ReleaseError(f"Glyph Forge executable is missing from {bundle}")
    return executable


def _run_smoke_command(executable: Path, arguments: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [str(executable), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReleaseError(
            f"Could not execute bundle smoke command: {arguments}"
        ) from exc
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise ReleaseError(
            f"Bundle command {' '.join(arguments)} failed ({result.returncode}): {details}"
        )
    return {
        "command": arguments,
        "seconds": round(elapsed, 3),
        "stdout_bytes": len(result.stdout.encode("utf-8")),
    }


def smoke_bundle(bundle: Path, *, version: str | None = None) -> list[dict[str, Any]]:
    """Exercise startup, rendering, and Studio resources in a frozen bundle."""

    executable = bundle_executable(bundle)
    expected_version = version or project_version()
    started = time.perf_counter()
    try:
        version_result = subprocess.run(
            [str(executable), "version", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReleaseError("Could not execute bundle version check") from exc
    version_elapsed = time.perf_counter() - started
    if version_result.returncode != 0:
        raise ReleaseError(
            version_result.stderr.strip() or "Bundle version check failed"
        )
    try:
        report = json.loads(version_result.stdout)
    except json.JSONDecodeError as exc:
        raise ReleaseError("Bundle version command did not return JSON") from exc
    if report.get("glyph_forge") != expected_version:
        raise ReleaseError(
            f"Bundle reports {report.get('glyph_forge')!r}, expected {expected_version!r}"
        )
    results = [
        {
            "command": ["version", "--json"],
            "seconds": round(version_elapsed, 3),
            "stdout_bytes": len(version_result.stdout.encode("utf-8")),
        },
        _run_smoke_command(executable, ["demo", "--width", "24", "--mode", "braille"]),
        _run_smoke_command(executable, ["studio", "--no-open", "--duration", "0.05"]),
    ]
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("version", help="Print the canonical package version.")
    verify = commands.add_parser("verify-tag", help="Validate a vVERSION tag.")
    verify.add_argument("--tag", required=True)

    archive = commands.add_parser("archive", help="Archive one portable bundle.")
    archive.add_argument("--source", type=Path, required=True)
    archive.add_argument("--output-directory", type=Path, required=True)
    archive.add_argument("--platform", required=True)
    archive.add_argument("--epoch", type=int)

    compare = commands.add_parser("compare", help="Compare reproducible builds.")
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)

    normalize = commands.add_parser(
        "normalize-sdist", help="Normalize source-distribution archives."
    )
    normalize.add_argument("archives", type=Path, nargs="+")
    normalize.add_argument("--epoch", type=int)

    checksums = commands.add_parser("checksums", help="Write SHA256SUMS.")
    checksums.add_argument("--directory", type=Path, required=True)
    checksums.add_argument("--output", type=Path, required=True)

    smoke = commands.add_parser("smoke", help="Exercise a frozen bundle.")
    smoke.add_argument("--bundle", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run release helper commands with concise, actionable failures."""

    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "version":
            print(project_version())
        elif arguments.command == "verify-tag":
            print(verify_tag(arguments.tag))
        elif arguments.command == "archive":
            output = archive_bundle(
                arguments.source,
                arguments.output_directory,
                arguments.platform,
                epoch=arguments.epoch,
            )
            print(f"{output}  sha256:{sha256(output)}")
        elif arguments.command == "compare":
            inventory = compare_directories(arguments.first, arguments.second)
            print(f"reproducible files: {len(inventory)}")
        elif arguments.command == "normalize-sdist":
            for archive in arguments.archives:
                print(normalize_sdist(archive, epoch=arguments.epoch))
        elif arguments.command == "checksums":
            checksums = write_checksums(arguments.directory, arguments.output)
            print(f"checksummed assets: {len(checksums)}")
        elif arguments.command == "smoke":
            print(json.dumps(smoke_bundle(arguments.bundle), indent=2))
        else:  # pragma: no cover - argparse guarantees a known command
            raise ReleaseError(f"Unknown release command: {arguments.command}")
    except (OSError, ReleaseError, ValueError) as exc:
        print(f"release error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Batch scanning and processing for video files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Set

from ..streaming.engine import UnifiedStreamConfig, UnifiedStreamEngine
from ..streaming.naming import VIDEO_EXTS

DEFAULT_EXCLUDE_DIRS = {
    "/proc",
    "/sys",
    "/dev",
    "/run",
    "/tmp",
    "/var/tmp",
    "/var/cache",
}
DEFAULT_EXCLUDE_NAMES = {
    ".git",
    "__pycache__",
    ".cache",
    ".venv",
    "venv",
    "node_modules",
}


def _normalize_dir(path: str) -> str:
    return os.path.abspath(path.rstrip(os.sep))


def _is_excluded_dir(path: str, exclude_dirs: object) -> bool:
    path = _normalize_dir(path)
    try:
        iterator = iter(exclude_dirs)  # type: ignore[arg-type]
    except TypeError:
        return False
    for exc in iterator:
        if not isinstance(exc, str):
            continue
        exc_norm = _normalize_dir(exc)
        if path == exc_norm or path.startswith(exc_norm + os.sep):
            return True
    return False


def iter_video_files(
    roots: Sequence[str],
    include_exts: Optional[Set[str]] = None,
    exclude_dirs: Optional[Set[str]] = None,
    exclude_names: Optional[Set[str]] = None,
    follow_links: bool = False,
    on_error: Optional[callable] = None,
) -> Iterator[Path]:
    """Yield video files from roots with stable ordering and filters."""
    include = {ext.lower() for ext in VIDEO_EXTS} if include_exts is None else set(include_exts)
    excludes = set(DEFAULT_EXCLUDE_DIRS) if exclude_dirs is None else set(exclude_dirs)
    exclude_names = set(DEFAULT_EXCLUDE_NAMES) if exclude_names is None else set(exclude_names)

    for root in sorted({str(r) for r in roots}):
        if not os.path.exists(root):
            if on_error:
                on_error(OSError(f"Root does not exist: {root}"))
            continue
        for dirpath, dirnames, filenames in os.walk(
            root,
            topdown=True,
            followlinks=follow_links,
            onerror=on_error,
        ):
            if _is_excluded_dir(dirpath, excludes):
                dirnames[:] = []
                continue
            dirnames[:] = [
                name
                for name in sorted(dirnames)
                if name not in exclude_names
                and not _is_excluded_dir(os.path.join(dirpath, name), excludes)
            ]
            for filename in sorted(filenames):
                ext = Path(filename).suffix.lower()
                if ext in include:
                    yield Path(dirpath) / filename


@dataclass
class BatchState:
    processed: Set[str]
    processed_log: Path
    failed_log: Path

    @classmethod
    def load(cls, state_dir: Path, reset: bool = False) -> "BatchState":
        state_dir.mkdir(parents=True, exist_ok=True)
        processed_log = state_dir / "processed.txt"
        failed_log = state_dir / "failed.txt"
        if reset:
            if processed_log.exists():
                processed_log.unlink()
            if failed_log.exists():
                failed_log.unlink()
        processed: Set[str] = set()
        if processed_log.exists():
            processed = {
                line.strip()
                for line in processed_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        return cls(processed=processed, processed_log=processed_log, failed_log=failed_log)

    def mark_processed(self, path: str) -> None:
        if path in self.processed:
            return
        self.processed.add(path)
        with self.processed_log.open("a", encoding="utf-8") as handle:
            handle.write(path + "\n")

    def mark_failed(self, path: str, reason: str) -> None:
        with self.failed_log.open("a", encoding="utf-8") as handle:
            handle.write(f"{path}\t{reason}\n")


def process_videos(
    videos: Iterable[Path],
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    resolution: str = "720p",
    fps: int = 30,
    mode: str = "gradient",
    color: str = "ansi256",
    backend: str = "auto",
    buffer_seconds: float = 30.0,
    prebuffer_seconds: float = 5.0,
    max_duration: Optional[float] = None,
    mux_audio: bool = True,
    state: Optional[BatchState] = None,
    resume: bool = True,
    limit: Optional[int] = None,
    dry_run: bool = False,
    on_progress: Optional[callable] = None,
) -> tuple[int, int]:
    """Process a set of video files into glyph renders."""
    processed = 0
    failed = 0
    if isinstance(videos, (str, os.PathLike, Path)):
        iterable: Iterable[Path | str] = [videos]
    else:
        try:
            iter(videos)
            iterable = videos
        except TypeError:
            return processed, failed

    for path in iterable:
        path_str = str(path)
        if state and resume and path_str in state.processed:
            continue
        if limit is not None and processed + failed >= limit:
            break
        if dry_run:
            if on_progress:
                on_progress(path, "dry_run", None)
            continue
        try:
            config = UnifiedStreamConfig(
                source=path_str,
                resolution=resolution,
                target_fps=int(fps),
                render_mode=mode,
                color_mode=color,
                audio_enabled=True,
                mux_audio=mux_audio,
                record_enabled=True,
                record_path=None,
                output_dir=output_dir,
                overwrite_output=overwrite,
                render_then_play=True,
                play_after_render=False,
                show_metrics=False,
                buffer_seconds=buffer_seconds,
                prebuffer_seconds=prebuffer_seconds,
                frame_backend=backend,
                max_duration_seconds=max_duration,
            )
            engine = UnifiedStreamEngine(config)
            engine.run()
            if state:
                state.mark_processed(path_str)
            processed += 1
            if on_progress:
                on_progress(path, "processed", None)
        except Exception as exc:
            failed += 1
            if state:
                state.mark_failed(path_str, str(exc))
            if on_progress:
                on_progress(path, "failed", exc)
    return processed, failed

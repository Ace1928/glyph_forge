"""Tests for batch scanning and processing utilities."""

from __future__ import annotations

from pathlib import Path

from glyph_forge.services.batch import BatchState, iter_video_files


def test_iter_video_files_filters(tmp_path: Path) -> None:
    (tmp_path / "movie.mp4").write_bytes(b"video")
    (tmp_path / "clip.MKV").write_bytes(b"video")
    (tmp_path / "notes.txt").write_text("nope", encoding="utf-8")

    exclude_dir = tmp_path / "node_modules"
    exclude_dir.mkdir()
    (exclude_dir / "skip.mp4").write_bytes(b"video")

    results = list(
        iter_video_files(
            [str(tmp_path)],
            include_exts={".mp4", ".mkv"},
            exclude_dirs=set(),
            exclude_names={"node_modules"},
        )
    )
    names = sorted(path.name for path in results)
    assert names == ["clip.MKV", "movie.mp4"]


def test_batch_state_persists(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state = BatchState.load(state_dir)
    state.mark_processed("/tmp/example.mp4")

    reloaded = BatchState.load(state_dir)
    assert "/tmp/example.mp4" in reloaded.processed

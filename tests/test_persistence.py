"""Crash-safe shared persistence behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from glyph_forge import persistence
from glyph_forge.persistence import (
    AtomicWriteError,
    atomic_write_bytes,
    atomic_write_text,
)


def test_atomic_text_replaces_existing_content_without_temp_files(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "result.txt"
    destination.parent.mkdir()
    destination.write_text("old", encoding="utf-8")

    result = atomic_write_text(destination, "new ✓")

    assert result == destination
    assert destination.read_text(encoding="utf-8") == "new ✓"
    assert not list(destination.parent.glob(".*.tmp"))


def test_atomic_binary_write_preserves_destination_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "result.bin"
    destination.write_bytes(b"old")
    monkeypatch.setattr(
        persistence.os,
        "replace",
        lambda *_: (_ for _ in ()).throw(OSError("interrupted")),
    )

    with pytest.raises(AtomicWriteError, match="interrupted"):
        atomic_write_bytes(destination, b"new")

    assert destination.read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_writers_reject_the_wrong_payload_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="bytes"):
        atomic_write_bytes(tmp_path / "bad", "text")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="string"):
        atomic_write_text(tmp_path / "bad", b"bytes")  # type: ignore[arg-type]

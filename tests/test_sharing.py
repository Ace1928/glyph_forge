"""Tests for bounded, file-backed ephemeral sharing."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from glyph_forge.sharing import (
    EphemeralShareStore,
    ShareError,
    ShareLimitError,
    ShareUnavailableError,
    media_type_for,
    safe_filename,
)


class Clock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def token_factory() -> Callable[[], str]:
    count = 0

    def make_token() -> str:
        nonlocal count
        count += 1
        return f"token_{count:020d}"

    return make_token


def make_store(**overrides: object) -> EphemeralShareStore:
    options: dict[str, Any] = {
        "clock": Clock(),
        "token_factory": token_factory(),
        "default_ttl": 10.0,
        "max_ttl": 20.0,
        "max_items": 3,
        "max_memory_bytes": 16,
        "max_upload_bytes": 8,
    }
    options.update(overrides)
    return EphemeralShareStore(**options)


def test_memory_share_expires_and_releases_its_budget() -> None:
    clock = Clock()
    store = make_store(clock=clock)
    asset = store.publish_bytes(b"PNG", "frame.png", media_type="image/png")

    assert store.get(asset.token) == asset
    assert store.memory_bytes == 3
    assert len(store) == 1

    clock.now += 11
    with pytest.raises(ShareUnavailableError, match="expired"):
        store.get(asset.token)
    assert store.memory_bytes == 0
    assert len(store) == 0


def test_expiry_uses_a_monotonic_deadline_when_wall_time_changes() -> None:
    clock = Clock()
    wall_clock = Clock(10_000.0)
    store = make_store(clock=clock, wall_clock=wall_clock)
    asset = store.publish_bytes(b"PNG", "frame.png")
    assert asset.created_at == 10_000.0
    assert asset.expires_at == 10_010.0

    wall_clock.now = 1.0
    clock.now += 11

    with pytest.raises(ShareUnavailableError):
        store.get(asset.token)


def test_store_evicts_old_memory_shares_to_stay_bounded() -> None:
    store = make_store(max_memory_bytes=6, max_upload_bytes=4)
    first = store.publish_bytes(b"1234", "first.png")
    second = store.publish_bytes(b"5678", "second.png")

    with pytest.raises(ShareUnavailableError):
        store.get(first.token)
    assert store.get(second.token).data == b"5678"
    assert store.memory_bytes == 4


def test_store_evicts_oldest_item_and_supports_revocation() -> None:
    store = make_store(max_items=1)
    first = store.publish_bytes(b"1", "first.png")
    second = store.publish_bytes(b"2", "second.png")

    with pytest.raises(ShareUnavailableError):
        store.get(first.token)
    store.discard(second.token)
    assert len(store) == 0


def test_browser_upload_and_ttl_limits_are_enforced() -> None:
    store = make_store()
    existing = store.publish_bytes(b"old", "old.png")

    with pytest.raises(ShareLimitError, match="upload limit"):
        store.publish_bytes(b"123456789", "large.png")
    with pytest.raises(ValueError, match="TTL"):
        store.publish_bytes(b"x", "long.png", ttl=21)
    assert store.get(existing.token).data == b"old"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"default_ttl": 0},
        {"default_ttl": 2, "max_ttl": 1},
        {"max_items": 0},
        {"max_memory_bytes": 0},
        {"max_upload_bytes": 9, "max_memory_bytes": 8},
    ],
)
def test_invalid_store_limits_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_store(**kwargs)


def test_file_shares_are_not_loaded_and_detect_modification(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"0123456789")
    store = make_store()
    asset = store.publish_file(source)

    assert asset.data is None
    assert asset.size == 10
    assert store.memory_bytes == 0
    with asset.open_file() as stream:
        assert stream.read() == b"0123456789"

    source.write_bytes(b"changed")
    with pytest.raises(ShareUnavailableError, match="changed"):
        store.get(asset.token)
    assert len(store) == 0


def test_only_regular_existing_files_can_be_shared(tmp_path: Path) -> None:
    store = make_store()

    with pytest.raises(ShareError):
        store.publish_file(tmp_path)
    with pytest.raises(ShareError, match="inspect"):
        store.publish_file(tmp_path / "missing.mp4")


def test_token_factory_must_return_safe_unique_tokens() -> None:
    unsafe = make_store(token_factory=lambda: "not safe")
    with pytest.raises(ShareError, match="unsafe"):
        unsafe.publish_bytes(b"x", "x.png")

    duplicate = make_store(token_factory=lambda: "x" * 24)
    duplicate.publish_bytes(b"x", "x.png")
    with pytest.raises(ShareError, match="unique"):
        duplicate.publish_bytes(b"y", "y.png")


def test_filenames_and_media_types_are_header_safe() -> None:
    assert safe_filename("../bad\r\n/name?.mp4") == "name_.mp4"
    assert safe_filename("...") == "glyph-forge-output"
    assert media_type_for("clip.mp4") == "video/mp4"
    assert media_type_for("clip", "text/plain; charset=utf-8") == (
        "application/octet-stream"
    )


def test_clear_releases_all_memory() -> None:
    store = make_store()
    store.publish_bytes(b"one", "one.png")
    store.publish_bytes(b"two", "two.png")

    store.clear()

    assert len(store) == 0
    assert store.memory_bytes == 0

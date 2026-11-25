"""Tests for share link encoding/decoding."""

from __future__ import annotations

from pathlib import Path

import pytest

from glyph_forge.cli.share_utils import decode_share_link, encode_share_link, load_share_link
from glyph_forge.exceptions import GlyphForgeInputError


def test_share_link_roundtrip_text() -> None:
    payload = "Glyph Forge".encode("utf-8")
    link = encode_share_link(payload, "txt", "demo.txt", source="unit-test")
    meta, data = decode_share_link(link)
    assert data == payload
    assert meta["format"] == "txt"


def test_share_link_roundtrip_binary_compressed() -> None:
    payload = b"x" * 4096
    link = encode_share_link(payload, "bin", "blob.bin", source="unit-test")
    meta, data = decode_share_link(link)
    assert data == payload
    assert meta["compression"] in (None, "gzip")


def test_load_share_link_from_file(tmp_path: Path) -> None:
    payload = b"hello"
    link = encode_share_link(payload, "txt", "demo.txt")
    link_path = tmp_path / "demo.gflink"
    link_path.write_text(link, encoding="utf-8")

    meta, data = load_share_link(str(link_path))
    assert data == payload
    assert meta["filename"] == "demo.txt"


@pytest.mark.parametrize("link", ["glyphforge://not-base64!", "glyphforge://e30="])
def test_malformed_links_fail_with_public_input_error(link: str) -> None:
    with pytest.raises(GlyphForgeInputError):
        decode_share_link(link)


def test_oversized_payload_is_refused_before_encoding(monkeypatch) -> None:
    monkeypatch.setattr("glyph_forge.cli.share_utils.MAX_SHARE_BYTES", 8)

    with pytest.raises(GlyphForgeInputError, match="safety limit"):
        encode_share_link(b"123456789", "bin", "payload.bin")


def test_filename_cannot_escape_envelope() -> None:
    link = encode_share_link(b"safe", "txt", "../demo.txt")

    payload, _ = decode_share_link(link)

    assert payload["filename"] == "demo.txt"

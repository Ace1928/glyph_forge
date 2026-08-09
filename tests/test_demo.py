"""Tests for the self-contained demo showcase."""

from __future__ import annotations

import http.server
import json
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from glyph_forge.cli import app
from glyph_forge.demo import run_demo

_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x08\x00\x00\x00\x08"
    b"\x08\x02\x00\x00\x00Km)\xdc\x00\x00\x00\x14IDATx\x9cc\xac\xd08\xc1"
    b"\x80\r0a\x15\x1d\xb4\x12\x00\x1d7\x01x+Z\xd9?\x00\x00\x00\x00IEND"
    b"\xaeB`\x82"
)


class _PNGHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(_PNG_BYTES)))
        self.end_headers()
        self.wfile.write(_PNG_BYTES)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture()
def png_server() -> str:  # type: ignore[misc]  # pragma: no cover
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _PNGHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("glyph_forge.demo._FETCH_RETRIES", 1)
    monkeypatch.setattr("glyph_forge.demo._FETCH_TIMEOUT", 2.0)


def test_demo_offline_writes_artifacts(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "demo",
            "--mode",
            "half-block",
            "--width",
            "46",
            "--no-color",
            "--offline",
            "--no-media",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "MEME WALL" in result.output
    assert "Glyph Forge · half-block" in result.output
    assert "FORGE" in result.output
    assert "crafted offline" in result.output
    pngs = list(tmp_path.glob("*.png"))
    txts = list(tmp_path.glob("*.txt"))
    assert len(pngs) == 3
    assert len(txts) >= 4
    assert (tmp_path / "meme-drake.png").is_file()
    assert (tmp_path / "mode-half-block.txt").is_file()


def test_demo_output_flag(tmp_path: Path) -> None:
    target = tmp_path / "show.txt"
    result = CliRunner().invoke(
        app,
        [
            "demo",
            "--mode",
            "braille",
            "--width",
            "20",
            "--no-color",
            "--offline",
            "--no-media",
            "--output",
            str(target),
        ],
    )

    assert result.exit_code == 0, result.output
    assert target.is_file()
    content = target.read_text(encoding="utf-8")
    assert "MEME WALL" in content


def test_demo_rejects_unknown_mode() -> None:
    result = CliRunner().invoke(
        app, ["demo", "--mode", "vaporwave", "--offline", "--no-media"]
    )

    assert result.exit_code == 2
    assert "Unknown render mode 'vaporwave'" in result.output


def test_demo_fetches_memes_from_local_server(
    tmp_path: Path, png_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("glyph_forge.demo._MEME_BASE", png_server)
    monkeypatch.setattr("glyph_forge.demo._THUMB_BASE", png_server)
    monkeypatch.setattr(
        "glyph_forge.demo._CACHE_ROOT", tmp_path / "cache" / "glyph_forge" / "demo"
    )

    result = CliRunner().invoke(
        app,
        [
            "demo",
            "--mode",
            "braille",
            "--width",
            "40",
            "--no-color",
            "--no-media",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "downloaded live" in result.output
    assert "3 assets fetched" in result.output
    assert len(list((tmp_path / "cache" / "glyph_forge" / "demo").glob("*.bin"))) == 3


def test_run_demo_result(png_server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("glyph_forge.demo._MEME_BASE", png_server)
    monkeypatch.setattr("glyph_forge.demo._THUMB_BASE", png_server)

    result = run_demo(mode="braille", width=40, color=False, media=False)

    assert result.stats.renders >= 4
    assert result.stats.assets_fetched == 3
    assert result.stats.assets_fallback == 0
    assert "Glyph Forge · braille" in result.text
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["stats"]["scenes"] == 6

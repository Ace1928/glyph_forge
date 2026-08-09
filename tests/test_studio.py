"""Tests for the local dependency-free browser studio server."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest
from typer.testing import CliRunner

from glyph_forge.cli import app
from glyph_forge.studio import StudioError, StudioServer


def test_studio_serves_assets_with_security_headers() -> None:
    with StudioServer(port=0) as server:
        with urllib.request.urlopen(server.url, timeout=2) as response:
            html = response.read().decode("utf-8")
            headers = response.headers

    assert "Glyph Forge Studio" in html
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "object-src 'none'" in headers["Content-Security-Policy"]
    assert headers["Cache-Control"] == "no-store"


def test_studio_does_not_serve_files_outside_its_asset_root() -> None:
    with StudioServer(port=0) as server:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"{server.url}../../pyproject.toml", timeout=2)

    assert error.value.code == 404


def test_network_bind_requires_explicit_consent() -> None:
    with pytest.raises(StudioError, match="allow-network"):
        StudioServer("0.0.0.0", port=0)


def test_studio_cli_can_run_headlessly_for_automation() -> None:
    result = CliRunner().invoke(
        app,
        ["studio", "--no-open", "--duration", "0.02"],
    )

    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:" in result.output

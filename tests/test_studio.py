"""Tests for the local dependency-free browser studio server."""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from rich.text import Text
from typer.testing import CliRunner

from glyph_forge.cli import app
from glyph_forge.studio import (
    StudioError,
    StudioServer,
    _format_url_host,
    _parse_byte_range,
)


def fetch_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=2) as response:
        return json.loads(response.read())


def studio_config(server: StudioServer) -> dict[str, object]:
    return fetch_json(f"{server.url}api/config")


def publish_png(
    server: StudioServer,
    payload: bytes = b"\x89PNG\r\n\x1a\nPNG data",
    *,
    origin: str | None = None,
    csrf: str | None = None,
    media_type: str = "image/png",
) -> urllib.request.Request:
    config = studio_config(server)
    request = urllib.request.Request(
        f"{server.url}api/share?name=forge.png",
        data=payload,
        method="POST",
        headers={
            "Content-Type": media_type,
            "Origin": origin or server.url.rstrip("/"),
            "X-Glyph-Forge-Token": csrf or str(config["csrf_token"]),
        },
    )
    return request


def test_studio_serves_assets_with_security_headers() -> None:
    with StudioServer(port=0) as server:
        with urllib.request.urlopen(server.url, timeout=2) as response:
            html = response.read().decode("utf-8")
            headers = response.headers

    assert "Glyph Forge Studio" in html
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "object-src 'none'" in headers["Content-Security-Policy"]
    assert "worker-src 'self'" in headers["Content-Security-Policy"]
    assert "camera=(self)" in headers["Permissions-Policy"]
    assert headers["Cross-Origin-Resource-Policy"] == "same-origin"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Cache-Control"] == "no-store"


def test_studio_config_reports_link_sharing_disabled_by_default() -> None:
    with StudioServer(port=0) as server:
        config = studio_config(server)
        request = urllib.request.Request(
            f"{server.url}api/share",
            data=b"PNG",
            method="POST",
            headers={"Content-Type": "image/png"},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=2)

    assert config["share_links"] is False
    assert config["csrf_token"] is None
    assert error.value.code == 403


def test_studio_does_not_serve_files_outside_its_asset_root() -> None:
    with StudioServer(port=0) as server:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"{server.url}../../pyproject.toml", timeout=2)

    assert error.value.code == 404


def test_network_bind_requires_explicit_consent() -> None:
    with pytest.raises(StudioError, match="allow-network"):
        StudioServer("0.0.0.0", port=0)


def test_browser_can_publish_and_range_read_a_bounded_png() -> None:
    with StudioServer(port=0, share_links=True, share_ttl=60) as server:
        config = studio_config(server)
        with urllib.request.urlopen(publish_png(server), timeout=2) as response:
            result = json.loads(response.read())
            assert response.status == 201

        with urllib.request.urlopen(str(result["url"]), timeout=2) as response:
            assert response.read() == b"\x89PNG\r\n\x1a\nPNG data"
            assert response.headers["Accept-Ranges"] == "bytes"
            assert response.headers["Content-Disposition"].startswith("inline;")
            assert "sandbox" in response.headers["Content-Security-Policy"]

        partial = urllib.request.Request(
            str(result["url"]), headers={"Range": "bytes=1-3"}
        )
        with urllib.request.urlopen(partial, timeout=2) as response:
            assert response.status == 206
            assert response.read() == b"PNG"
            assert response.headers["Content-Range"] == "bytes 1-3/16"

    assert config["share_links"] is True
    assert config["max_upload_bytes"] == 16 * 1024 * 1024
    assert config["public_base_url"] == server.public_url


def test_browser_publish_rejects_wrong_origin_type_and_oversize() -> None:
    with StudioServer(port=0, share_links=True, max_upload_bytes=4) as server:
        with pytest.raises(urllib.error.HTTPError) as wrong_origin:
            urllib.request.urlopen(
                publish_png(server, origin="http://attacker.invalid"), timeout=2
            )
        with pytest.raises(urllib.error.HTTPError) as wrong_type:
            urllib.request.urlopen(
                publish_png(server, b"x", media_type="text/plain"), timeout=2
            )
        with pytest.raises(urllib.error.HTTPError) as too_large:
            urllib.request.urlopen(publish_png(server, b"12345"), timeout=2)

    assert wrong_origin.value.code == 403
    assert wrong_type.value.code == 415
    assert too_large.value.code == 413


def test_browser_publish_rejects_empty_and_falsely_labelled_png() -> None:
    with StudioServer(port=0, share_links=True) as server:
        with pytest.raises(urllib.error.HTTPError) as empty:
            urllib.request.urlopen(publish_png(server, b""), timeout=2)
        with pytest.raises(urllib.error.HTTPError) as invalid:
            urllib.request.urlopen(publish_png(server, b"not a png"), timeout=2)

    assert empty.value.code == 400
    assert invalid.value.code == 400


def test_file_only_server_does_not_expose_browser_upload_api(tmp_path: Path) -> None:
    source = tmp_path / "render.mp4"
    source.write_bytes(b"video")
    server = StudioServer(port=0, share_links=True, browser_shares=False)
    publication = server.publish_file(source)

    with server:
        assert studio_config(server)["share_links"] is False
        with urllib.request.urlopen(publication.url, timeout=2) as response:
            assert response.read() == b"video"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(publish_png(server), timeout=2)

    assert error.value.code == 403


def test_closed_server_rejects_new_publications(tmp_path: Path) -> None:
    source = tmp_path / "render.mp4"
    source.write_bytes(b"video")
    server = StudioServer(port=0, share_links=True)
    server.close()

    with pytest.raises(StudioError, match="closed"):
        server.publish_file(source)


def test_file_link_streams_ranges_and_head_without_copying(tmp_path: Path) -> None:
    source = tmp_path / "music video.mp4"
    source.write_bytes(bytes(range(64)))
    server = StudioServer(port=0, share_links=True)
    publication = server.publish_file(source)

    with server:
        request = urllib.request.Request(
            publication.url, headers={"Range": "bytes=10-19"}
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            assert response.status == 206
            assert response.read() == bytes(range(10, 20))
            assert response.headers["Content-Range"] == "bytes 10-19/64"

        suffix = urllib.request.Request(publication.url, headers={"Range": "bytes=-4"})
        with urllib.request.urlopen(suffix, timeout=2) as response:
            assert response.read() == bytes(range(60, 64))

        head = urllib.request.Request(publication.url, method="HEAD")
        with urllib.request.urlopen(head, timeout=2) as response:
            assert response.status == 200
            assert response.headers["Content-Length"] == "64"
            assert response.read() == b""

        invalid = urllib.request.Request(
            publication.url, headers={"Range": "bytes=99-100"}
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(invalid, timeout=2)
        assert error.value.code == 416
        assert error.value.headers["Content-Range"] == "bytes */64"


def test_file_link_is_revoked_if_source_changes(tmp_path: Path) -> None:
    source = tmp_path / "render.mp4"
    source.write_bytes(b"original")
    server = StudioServer(port=0, share_links=True)
    publication = server.publish_file(source)
    source.write_bytes(b"different")

    with server, pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(publication.url, timeout=2)

    assert error.value.code == 404


def test_untrusted_host_header_cannot_read_api_configuration() -> None:
    with StudioServer(port=0, share_links=True) as server:
        request = urllib.request.Request(
            f"{server.url}api/config", headers={"Host": "attacker.invalid"}
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=2)

    assert error.value.code == 403


@pytest.mark.parametrize(
    ("header", "size", "expected"),
    [
        (None, 10, None),
        ("bytes=2-", 10, (2, 9)),
        ("bytes=-20", 10, (0, 9)),
        ("bytes=2-99", 10, (2, 9)),
    ],
)
def test_byte_range_parser(
    header: str | None, size: int, expected: tuple[int, int] | None
) -> None:
    assert _parse_byte_range(header, size) == expected


@pytest.mark.parametrize(
    "header", ["items=1-2", "bytes=", "bytes=2-1", "bytes=0-1,3-4"]
)
def test_byte_range_parser_rejects_unsafe_forms(header: str) -> None:
    with pytest.raises(ValueError):
        _parse_byte_range(header, 10)


def test_ipv6_hosts_are_bracketed_for_urls() -> None:
    assert _format_url_host("::1") == "[::1]"
    assert _format_url_host("fe80::1%eth0") == "[fe80::1%25eth0]"
    assert _format_url_host("studio.local") == "studio.local"


def test_advertised_host_is_validated() -> None:
    with pytest.raises(ValueError, match="hostname or IP"):
        StudioServer(advertise_host="https://example.test")


@pytest.mark.parametrize("connections", [0, 257])
def test_connection_limit_is_bounded(connections: int) -> None:
    with pytest.raises(ValueError, match="max_connections"):
        StudioServer(max_connections=connections)


def test_browser_assets_include_opt_in_publish_controls() -> None:
    with StudioServer(port=0) as server:
        with urllib.request.urlopen(server.url, timeout=2) as response:
            html = response.read().decode("utf-8")
        with urllib.request.urlopen(f"{server.url}studio.js", timeout=2) as response:
            javascript = response.read().decode("utf-8")

    assert 'id="publishButton"' in html
    assert 'fetch(studioEndpoint("api/config")' in javascript
    assert "studioEndpoint(`api/share?name=" in javascript


def test_browser_assets_form_an_installable_offline_app_shell() -> None:
    with StudioServer(port=0) as server:
        assets: dict[str, tuple[str, bytes]] = {}
        for name in (
            "manifest.webmanifest",
            "service-worker.js",
            "project-contract.js",
            "icon.svg",
            "icon-192.png",
            "icon-512.png",
            "apple-touch-icon.png",
            "social-card.png",
        ):
            with urllib.request.urlopen(f"{server.url}{name}", timeout=2) as response:
                assets[name] = (response.headers.get_content_type(), response.read())

    manifest = json.loads(assets["manifest.webmanifest"][1])
    assert assets["manifest.webmanifest"][0] == "application/manifest+json"
    assert assets["service-worker.js"][0] == "text/javascript"
    assert assets["project-contract.js"][0] == "text/javascript"
    assert manifest["display"] == "standalone"
    assert manifest["start_url"] == "./"
    assert {icon["sizes"] for icon in manifest["icons"]} >= {"192x192", "512x512"}
    assert manifest["file_handlers"][0]["action"] == "./"
    assert assets["icon-192.png"][1].startswith(b"\x89PNG\r\n\x1a\n")
    assert assets["icon-512.png"][1].startswith(b"\x89PNG\r\n\x1a\n")
    assert assets["apple-touch-icon.png"][1].startswith(b"\x89PNG\r\n\x1a\n")
    assert assets["social-card.png"][1].startswith(b"\x89PNG\r\n\x1a\n")
    worker = assets["service-worker.js"][1].decode("utf-8")
    assert 'url.pathname.includes("/api/")' in worker
    assert 'url.pathname.includes("/s/")' in worker


def test_browser_assets_expose_full_fidelity_and_recording_controls() -> None:
    with StudioServer(port=0) as server:
        with urllib.request.urlopen(server.url, timeout=2) as response:
            html = response.read().decode("utf-8")
        with urllib.request.urlopen(f"{server.url}studio.js", timeout=2) as response:
            javascript = response.read().decode("utf-8")

    for mode in ("glyph", "edge", "braille", "half-block", "quadrant"):
        assert f'value="{mode}"' in html
    for control in (
        "textSourceInput",
        "audioToggle",
        "recordButton",
        "fullscreenButton",
        "installButton",
    ):
        assert f'id="{control}"' in html
    assert '<script type="module" src="studio.js"></script>' in html
    assert 'from "./studio-renderers.js"' in javascript
    assert "requestVideoFrameCallback" in javascript
    assert "elements.canvas.captureStream(frameRate)" in javascript
    assert (
        "new MediaStream([...canvasStream.getVideoTracks(), ...audioTracks])"
        in javascript
    )
    assert "new MediaRecorder(stream" in javascript
    assert "requestFullscreen" in javascript


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_browser_javascript_parses() -> None:
    with StudioServer(port=0) as server:
        assets = []
        for name in (
            "studio.js",
            "studio-renderers.js",
            "project-contract.js",
            "service-worker.js",
        ):
            with urllib.request.urlopen(f"{server.url}{name}", timeout=2) as response:
                assets.append((name, response.read()))

    for name, javascript in assets:
        completed = subprocess.run(
            [
                shutil.which("node") or "node",
                "--input-type=module",
                "--check",
                "-",
            ],
            input=javascript,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, (
            f"{name}: {completed.stderr.decode('utf-8', 'replace')}"
        )


def test_studio_cli_can_run_headlessly_for_automation() -> None:
    result = CliRunner().invoke(
        app,
        ["studio", "--no-open", "--duration", "0.02"],
    )

    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:" in result.output


def test_share_cli_serves_one_file_headlessly(tmp_path: Path) -> None:
    source = tmp_path / "finished.mp4"
    source.write_bytes(b"video")
    result = CliRunner().invoke(
        app,
        ["share", str(source), "--duration", "0.02", "--ttl", "2"],
    )
    output = Text.from_ansi(result.output).plain

    assert result.exit_code == 0, output
    assert "Temporary share" in output
    assert "finished.mp4" in output
    assert "http://127.0.0.1:" in output
    assert "not copied or uploaded" in output


def test_share_publication_is_available_from_the_lazy_public_api() -> None:
    from glyph_forge import SharePublication

    assert SharePublication.__module__ == "glyph_forge.studio"

"""Secure local server for the dependency-free Glyph Forge browser studio."""

from __future__ import annotations

import functools
import ipaddress
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class StudioError(RuntimeError):
    """Raised when the local studio cannot be started safely."""


def _is_loopback(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _StudioHandler(SimpleHTTPRequestHandler):
    """Static-only request handler with browser security headers."""

    server_version = "GlyphForgeStudio/0.2"

    def __init__(self, *args: Any, quiet: bool = True, **kwargs: Any) -> None:
        self._quiet = quiet
        super().__init__(*args, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob: data:; "
            "media-src 'self' blob:; connect-src 'self'; "
            "style-src 'self'; script-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        if not self._quiet:
            super().log_message(format, *args)


class StudioServer:
    """Lifecycle wrapper around the local browser-studio HTTP server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        allow_network: bool = False,
        quiet: bool = True,
    ) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("Port must be between 0 and 65535")
        if not _is_loopback(host) and not allow_network:
            raise StudioError(
                "Binding the studio beyond this device requires --allow-network"
            )
        assets = Path(__file__).with_name("ui") / "web"
        if not (assets / "index.html").is_file():
            raise StudioError(f"Browser studio assets are missing: {assets}")
        handler = functools.partial(
            _StudioHandler,
            directory=str(assets),
            quiet=quiet,
        )
        try:
            self._server = ThreadingHTTPServer((host, port), handler)
        except OSError as exc:
            raise StudioError(f"Could not bind studio to {host}:{port}: {exc}") from exc
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None
        self.host = host
        self.port = int(self._server.server_address[1])

    @property
    def url(self) -> str:
        display_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{display_host}:{self.port}/"

    def start(self, *, open_browser: bool = False) -> "StudioServer":
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                kwargs={"poll_interval": 0.1},
                name="glyph-forge-studio",
                daemon=True,
            )
            self._thread.start()
        if open_browser:
            webbrowser.open(self.url, new=2)
        return self

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is None:
            self.start()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def close(self) -> None:
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=2)
            self._thread = None
        self._server.server_close()

    def __enter__(self) -> "StudioServer":
        return self.start()

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["StudioError", "StudioServer"]

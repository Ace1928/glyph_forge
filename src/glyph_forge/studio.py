"""Secure local server and ephemeral sharing surface for Glyph Forge Studio."""

from __future__ import annotations

import functools
import ipaddress
import json
import secrets
import socket
import threading
import webbrowser
from dataclasses import dataclass, field
from email.utils import formatdate
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import parse_qs, quote, urlsplit

from .sharing import (
    EphemeralShareStore,
    ShareAsset,
    ShareError,
    ShareLimitError,
    ShareUnavailableError,
)

_BROWSER_SHARE_TYPES = frozenset({"image/png"})
_STREAM_CHUNK_BYTES = 1024 * 1024
_SocketRequest = socket.socket | tuple[bytes, socket.socket]


class StudioError(RuntimeError):
    """Raised when the local studio cannot be started safely."""


@dataclass(frozen=True, slots=True)
class SharePublication:
    """Public details for one temporary capability link."""

    url: str
    filename: str
    media_type: str
    size: int
    expires_at: float


@dataclass(slots=True)
class _StudioContext:
    assets: Path
    shares: EphemeralShareStore | None
    browser_shares: bool = False
    allowed_hosts: set[str] = field(default_factory=set)
    public_base_url: str = ""
    csrf_token: str = field(default_factory=lambda: secrets.token_urlsafe(24))


def _strip_ip_brackets(host: str) -> str:
    value = host.strip()
    if value.startswith("[") and value.endswith("]"):
        return value[1:-1]
    return value


def _is_loopback(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(_strip_ip_brackets(host)).is_loopback
    except ValueError:
        return False


def _is_wildcard(host: str) -> bool:
    try:
        return ipaddress.ip_address(_strip_ip_brackets(host)).is_unspecified
    except ValueError:
        return False


def _validate_advertise_host(host: str) -> str:
    value = _strip_ip_brackets(host)
    if not value or any(character.isspace() for character in value):
        raise ValueError("Advertised host cannot be empty or contain whitespace")
    if any(character in value for character in "/?#@"):
        raise ValueError("Advertised host must be a hostname or IP address")
    try:
        ipaddress.ip_address(value)
        return value
    except ValueError:
        if ":" in value:
            raise ValueError("Advertised host is not a valid IPv6 address") from None
    try:
        ascii_host = value.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("Advertised hostname is invalid") from exc
    if len(ascii_host) > 253 or any(
        not label
        or len(label) > 63
        or label.startswith("-")
        or label.endswith("-")
        or not all(character.isalnum() or character == "-" for character in label)
        for label in ascii_host.rstrip(".").split(".")
    ):
        raise ValueError("Advertised hostname is invalid")
    return ascii_host


def _format_url_host(host: str) -> str:
    value = _strip_ip_brackets(host)
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return value
    if address.version == 6:
        return f"[{value.replace('%', '%25')}]"
    return value


def _http_url(host: str, port: int) -> str:
    return f"http://{_format_url_host(host)}:{port}/"


def _detect_lan_host(*, ipv6: bool = False) -> str:
    """Find a routable local address without sending network traffic."""

    family = socket.AF_INET6 if ipv6 else socket.AF_INET
    probe = "2001:db8::1" if ipv6 else "192.0.2.1"
    try:
        with socket.socket(family, socket.SOCK_DGRAM) as connection:
            connection.connect((probe, 9))
            candidate = str(connection.getsockname()[0])
            if not _is_loopback(candidate) and not _is_wildcard(candidate):
                return candidate
    except OSError:
        pass

    try:
        candidates = socket.getaddrinfo(
            socket.gethostname(), None, family, socket.SOCK_DGRAM
        )
    except OSError:
        candidates = []
    for result in candidates:
        candidate = str(result[4][0])
        if not _is_loopback(candidate) and not _is_wildcard(candidate):
            return candidate
    return "::1" if ipv6 else "127.0.0.1"


def _host_from_header(value: str) -> str | None:
    try:
        hostname = urlsplit(f"//{value}").hostname
    except ValueError:
        return None
    return hostname.replace("%25", "%") if hostname else None


def _parse_byte_range(value: str | None, size: int) -> tuple[int, int] | None:
    if value is None:
        return None
    if not value.startswith("bytes=") or "," in value or size <= 0:
        raise ValueError("Unsupported byte range")
    bounds = value[6:].strip()
    if bounds.count("-") != 1:
        raise ValueError("Malformed byte range")
    first, last = bounds.split("-", 1)
    try:
        if not first:
            suffix = int(last)
            if suffix <= 0:
                raise ValueError
            start = max(0, size - suffix)
            end = size - 1
        else:
            start = int(first)
            end = size - 1 if not last else min(int(last), size - 1)
            if start < 0 or start >= size or end < start:
                raise ValueError
    except ValueError as exc:
        raise ValueError("Malformed or unsatisfiable byte range") from exc
    return start, end


def _content_disposition(filename: str) -> str:
    fallback = "".join(
        character
        if 32 <= ord(character) < 127 and character not in {'"', "\\"}
        else "_"
        for character in filename
    )
    encoded = quote(filename, safe="")
    return f"inline; filename=\"{fallback}\"; filename*=UTF-8''{encoded}"


class _StudioHandler(SimpleHTTPRequestHandler):
    """Static studio plus tightly scoped, capability-addressed share endpoints."""

    server_version = "GlyphForgeStudio/0.3"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def __init__(
        self,
        *args: Any,
        quiet: bool = True,
        context: _StudioContext,
        **kwargs: Any,
    ) -> None:
        self._quiet = quiet
        self._context = context
        self._share_response = False
        super().__init__(*args, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        if self._share_response:
            self.send_header(
                "Content-Security-Policy",
                "sandbox; default-src 'none'; img-src data:; "
                "media-src 'self'; style-src 'unsafe-inline'; frame-ancestors 'none'",
            )
        else:
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

    def _request_host_allowed(self) -> bool:
        hostname = _host_from_header(self.headers.get("Host", ""))
        return bool(
            hostname and hostname.rstrip(".").casefold() in self._context.allowed_hosts
        )

    def _same_origin_request(self) -> bool:
        origin = self.headers.get("Origin", "")
        host = self.headers.get("Host", "")
        try:
            parts = urlsplit(origin)
        except ValueError:
            return False
        return (
            self._request_host_allowed()
            and parts.scheme == "http"
            and parts.netloc.casefold() == host.casefold()
            and parts.path in {"", "/"}
            and not parts.query
            and not parts.fragment
        )

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_plain_error(self, status: HTTPStatus, message: str) -> None:
        body = f"{message}\n".encode("utf-8")
        self._share_response = True
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _config(self) -> dict[str, object]:
        shares = self._context.shares
        enabled = shares is not None and self._context.browser_shares
        return {
            "share_links": enabled,
            "public_base_url": self._context.public_base_url if enabled else None,
            "default_ttl_seconds": shares.default_ttl if shares is not None else None,
            "max_upload_bytes": shares.max_upload_bytes if shares is not None else 0,
            "csrf_token": self._context.csrf_token if enabled else None,
        }

    def _share_token(self, path: str) -> str | None:
        pieces = path.split("/")
        if len(pieces) != 4 or pieces[1] != "s" or not pieces[2] or not pieces[3]:
            return None
        return pieces[2]

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/config":
            if not self._request_host_allowed():
                self._send_json(
                    HTTPStatus.FORBIDDEN, {"error": "Untrusted Host header"}
                )
                return
            self._send_json(HTTPStatus.OK, self._config())
            return
        token = self._share_token(path)
        if token is not None:
            self._serve_share(token, head_only=False)
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/config":
            if not self._request_host_allowed():
                self._send_json(
                    HTTPStatus.FORBIDDEN, {"error": "Untrusted Host header"}
                )
                return
            self._send_json(HTTPStatus.OK, self._config())
            return
        token = self._share_token(path)
        if token is not None:
            self._serve_share(token, head_only=True)
            return
        super().do_HEAD()

    def do_POST(self) -> None:
        parts = urlsplit(self.path)
        if parts.path != "/api/share":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})
            return
        shares = self._context.shares
        if shares is None or not self._context.browser_shares:
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Link sharing is disabled"})
            return
        csrf = self.headers.get("X-Glyph-Forge-Token", "")
        if not self._same_origin_request() or not secrets.compare_digest(
            csrf, self._context.csrf_token
        ):
            self._send_json(
                HTTPStatus.FORBIDDEN, {"error": "Request validation failed"}
            )
            return
        if self.headers.get("Transfer-Encoding"):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "Chunked browser uploads are not accepted"},
            )
            return
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            self._send_json(
                HTTPStatus.LENGTH_REQUIRED, {"error": "Content-Length required"}
            )
            return
        try:
            length = int(length_header)
        except ValueError:
            length = -1
        if length < 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length"})
            return
        if length == 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "PNG export is empty"})
            return
        if length > shares.max_upload_bytes:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "Browser export exceeds the configured upload limit"},
            )
            return
        media_type = self.headers.get_content_type().casefold()
        if media_type not in _BROWSER_SHARE_TYPES:
            self._send_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "Only PNG browser exports can be published"},
            )
            return
        payload = self.rfile.read(length)
        if len(payload) != length:
            self._send_json(
                HTTPStatus.BAD_REQUEST, {"error": "Incomplete request body"}
            )
            return
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid PNG export"})
            return
        filename = parse_qs(parts.query).get("name", ["glyph-forge.png"])[0]
        try:
            asset = shares.publish_bytes(payload, filename, media_type=media_type)
        except ShareLimitError as exc:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": str(exc)})
            return
        publication = self._publication(asset)
        self._send_json(
            HTTPStatus.CREATED,
            {
                "url": publication.url,
                "filename": publication.filename,
                "size": publication.size,
                "expires_at": publication.expires_at,
            },
        )

    def _publication(self, asset: ShareAsset) -> SharePublication:
        path = f"s/{asset.token}/{quote(asset.filename, safe='')}"
        return SharePublication(
            url=f"{self._context.public_base_url}{path}",
            filename=asset.filename,
            media_type=asset.media_type,
            size=asset.size,
            expires_at=asset.expires_at,
        )

    def _serve_share(self, token: str, *, head_only: bool) -> None:
        shares = self._context.shares
        if shares is None:
            self._send_plain_error(HTTPStatus.NOT_FOUND, "Unknown or expired link")
            return
        try:
            asset = shares.get(token)
        except ShareUnavailableError:
            self._send_plain_error(HTTPStatus.NOT_FOUND, "Unknown or expired link")
            return

        stream: BinaryIO | None = None
        if asset.is_file:
            try:
                stream = asset.open_file()
            except ShareUnavailableError:
                shares.discard(asset.token)
                self._send_plain_error(HTTPStatus.NOT_FOUND, "Unknown or expired link")
                return
        try:
            try:
                selected = _parse_byte_range(self.headers.get("Range"), asset.size)
            except ValueError:
                self._share_response = True
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{asset.size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            start, end = selected or (0, max(0, asset.size - 1))
            length = 0 if asset.size == 0 else end - start + 1
            self._share_response = True
            self.send_response(
                HTTPStatus.PARTIAL_CONTENT if selected else HTTPStatus.OK
            )
            self.send_header("Content-Type", asset.media_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", f'"{asset.token}"')
            self.send_header("Expires", formatdate(asset.expires_at, usegmt=True))
            self.send_header(
                "Content-Disposition", _content_disposition(asset.filename)
            )
            if selected:
                self.send_header("Content-Range", f"bytes {start}-{end}/{asset.size}")
            self.end_headers()
            if head_only or length == 0:
                return
            if stream is not None:
                self._stream_file(stream, start, length)
            else:
                payload = asset.data or b""
                self.wfile.write(payload[start : end + 1])
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            if stream is not None:
                stream.close()

    def _stream_file(self, stream: BinaryIO, start: int, length: int) -> None:
        self.wfile.flush()
        try:
            sent = self.connection.sendfile(stream, offset=start, count=length)
        except (AttributeError, NotImplementedError, ValueError):
            sent = 0
        if sent is None:
            sent = length
        remaining = length - sent
        if remaining <= 0:
            return
        stream.seek(start + sent)
        while remaining:
            chunk = stream.read(min(_STREAM_CHUNK_BYTES, remaining))
            if not chunk:
                break
            self.wfile.write(chunk)
            remaining -= len(chunk)


class _BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Threading server with a fixed upper bound on simultaneous requests."""

    request_queue_size = 64

    def __init__(self, *args: Any, max_connections: int = 32, **kwargs: Any) -> None:
        self._connection_slots = threading.BoundedSemaphore(max_connections)
        super().__init__(*args, **kwargs)

    def process_request(self, request: _SocketRequest, client_address: Any) -> None:
        if not self._connection_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connection_slots.release()
            raise

    def process_request_thread(
        self, request: _SocketRequest, client_address: Any
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connection_slots.release()


class _IPv6ThreadingHTTPServer(_BoundedThreadingHTTPServer):
    address_family = socket.AF_INET6


class StudioServer:
    """Lifecycle wrapper around the browser Studio and optional sharing server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        allow_network: bool = False,
        quiet: bool = True,
        share_links: bool = False,
        advertise_host: str | None = None,
        share_ttl: float = 3600.0,
        max_share_ttl: float = 86400.0,
        max_upload_bytes: int = 16 * 1024 * 1024,
        browser_shares: bool | None = None,
        max_connections: int = 32,
    ) -> None:
        host = _strip_ip_brackets(host)
        if not host:
            raise ValueError("Host cannot be empty")
        if not 0 <= port <= 65535:
            raise ValueError("Port must be between 0 and 65535")
        if max_connections < 1 or max_connections > 256:
            raise ValueError("max_connections must be between 1 and 256")
        if not _is_loopback(host) and not allow_network:
            raise StudioError(
                "Binding the studio beyond this device requires --allow-network"
            )
        advertised = (
            _validate_advertise_host(advertise_host)
            if advertise_host is not None
            else None
        )
        assets = Path(__file__).with_name("ui") / "web"
        if not (assets / "index.html").is_file():
            raise StudioError(f"Browser studio assets are missing: {assets}")
        shares = (
            EphemeralShareStore(
                default_ttl=share_ttl,
                max_ttl=max_share_ttl,
                max_upload_bytes=max_upload_bytes,
            )
            if share_links
            else None
        )
        context = _StudioContext(
            assets=assets,
            shares=shares,
            browser_shares=share_links if browser_shares is None else browser_shares,
        )
        handler = functools.partial(
            _StudioHandler,
            directory=str(assets),
            quiet=quiet,
            context=context,
        )
        server_class: type[_BoundedThreadingHTTPServer] = _BoundedThreadingHTTPServer
        try:
            if ipaddress.ip_address(host).version == 6:
                server_class = _IPv6ThreadingHTTPServer
        except ValueError:
            pass
        try:
            self._server = server_class(
                (host, port), handler, max_connections=max_connections
            )
        except OSError as exc:
            raise StudioError(f"Could not bind studio to {host}:{port}: {exc}") from exc
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None
        self._closed = False
        self.host = host
        self.port = int(self._server.server_address[1])
        self._context = context

        local_host = (
            "::1" if host == "::" else "127.0.0.1" if host == "0.0.0.0" else host
        )
        if advertised is not None:
            public_host = advertised
        elif _is_wildcard(host):
            public_host = _detect_lan_host(ipv6=host == "::")
        else:
            public_host = host
        self._local_url = _http_url(local_host, self.port)
        self._context.public_base_url = _http_url(public_host, self.port)
        allowed = {local_host, public_host, host, "localhost", socket.gethostname()}
        self._context.allowed_hosts = {
            value.rstrip(".").casefold()
            for value in allowed
            if value and not _is_wildcard(value)
        }

    @property
    def url(self) -> str:
        """Return the local browser URL."""

        return self._local_url

    @property
    def public_url(self) -> str:
        """Return the advertised URL used inside capability links."""

        return self._context.public_base_url

    @property
    def sharing_enabled(self) -> bool:
        return self._context.shares is not None

    def publish_file(
        self,
        path: str | Path,
        *,
        filename: str | None = None,
        media_type: str | None = None,
        ttl: float | None = None,
    ) -> SharePublication:
        """Publish exactly one file without copying it into memory."""

        if self._closed:
            raise StudioError("A closed Studio server cannot publish files")
        shares = self._context.shares
        if shares is None:
            raise StudioError("Link sharing is disabled for this server")
        try:
            asset = shares.publish_file(
                path,
                filename=filename,
                media_type=media_type,
                ttl=ttl,
            )
        except ShareError as exc:
            raise StudioError(str(exc)) from exc
        return self._publication(asset)

    def _publication(self, asset: ShareAsset) -> SharePublication:
        path = f"s/{asset.token}/{quote(asset.filename, safe='')}"
        return SharePublication(
            url=f"{self.public_url}{path}",
            filename=asset.filename,
            media_type=asset.media_type,
            size=asset.size,
            expires_at=asset.expires_at,
        )

    def start(self, *, open_browser: bool = False) -> StudioServer:
        if self._closed:
            raise StudioError("A closed Studio server cannot be restarted")
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
        if self._closed:
            return
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=2)
            self._thread = None
        self._server.server_close()
        if self._context.shares is not None:
            self._context.shares.clear()
        self._closed = True

    def __enter__(self) -> StudioServer:
        return self.start()

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["SharePublication", "StudioError", "StudioServer"]

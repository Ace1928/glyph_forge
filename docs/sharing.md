# Temporary sharing

Glyph Forge can share an output directly from the device that rendered it. The
feature is local-first: there is no account, upload provider, cloud storage, or
background daemon.

## Share one saved file

```bash
# Reachable only from this computer
glyph-forge share render.mp4

# Reachable from a trusted local network for 30 minutes
glyph-forge share render.mp4 --lan --ttl 1800
```

The command prints one random capability URL and keeps it active until Ctrl+C
or the TTL. `--open` opens that URL locally. With `--lan`, Glyph Forge binds to
all IPv4 interfaces and chooses the address embedded in the URL. Override an
incorrect choice on a VPN or multi-interface host:

```bash
glyph-forge share render.mp4 --lan --advertise-host studio.example.lan
```

The file stays at its original path. Glyph Forge serves only that resolved
regular file, checks its identity/size/modification time before each response,
and revokes the link if it changes. HEAD and single HTTP byte-range requests are
supported, allowing browsers and media players to seek. Compatible platforms
use the socket `sendfile` path; other platforms use a bounded 1 MiB streaming
buffer. Neither path makes a media-sized memory copy.

## Publish a Studio frame

Normal `glyph-forge studio` sessions have no publication endpoint. Enable it on
the same device with `--share-links`, or combine trusted-LAN listening and link
publication with `--lan`:

```bash
glyph-forge studio --share-links
glyph-forge studio --lan --share-ttl 1800
```

The **Copy temporary link** button renders the current output canvas to PNG,
publishes that snapshot to the local server, and copies its URL. It does not
publish the selected image/video, webcam feed, screen stream, or future frames.
Browser snapshots are restricted to PNG, 16 MiB each, 32 active items, 64 MiB
total retained memory, and a maximum one-day TTL. Old entries are evicted to
keep those bounds. The server also caps simultaneous request threads, keeping a
slow client or burst of connections from creating unbounded work.

The ordinary **Share file** button still uses the browser's Web Share API where
available, and **Copy style link** still contains settings without media.
The installable Studio's service worker explicitly bypasses `/api/` and `/s/`
routes, so temporary publications and capability responses are never placed in
its offline cache.

## Security model

A temporary URL is a bearer secret: anyone who knows it and can connect to the
server can read that output. Tokens contain 192 bits of cryptographic randomness
and are not listed by the server. Responses disable caching and MIME sniffing;
potentially active file types receive a restrictive sandbox policy. Browser
publication also requires an allowed Host header, a same-origin POST, and the
per-session CSRF token obtained by the Studio page.

Traffic is plain HTTP. Use LAN mode only with people and networks you trust, do
not forward its port to the Internet, and stop the command to reject new link
requests immediately. For remote public sharing, export the file and use a
provider whose authentication, encryption, retention, and abuse controls you
accept.

## Python API

```python
from glyph_forge import StudioServer

server = StudioServer(
    "127.0.0.1",
    share_links=True,
    share_ttl=900,
)
publication = server.publish_file("render.mp4")
print(publication.url, publication.size, publication.expires_at)

with server:
    server.wait()
```

`publish_file` is file-backed and returns a `SharePublication`. Applications
that bind a non-loopback address must pass `allow_network=True`; use
`advertise_host` when the bind address is a wildcard or is not the hostname
clients should receive.

# Security policy

## Supported versions

Security fixes target the current `0.3.x` line and `main`. Older releases may
receive a fix when practical but are not actively supported.

## Report a vulnerability

Do not open a public issue for a vulnerability that could put users at risk.
Prefer a [private GitHub security advisory](https://github.com/Ace1928/glyph_forge/security/advisories/new),
or send a private report to <lloyd.handyside@neuroforge.io> or
<ace1928@gmail.com>, with:

- affected version and platform;
- a minimal reproduction or proof of concept;
- expected impact and required preconditions;
- any mitigation already known;
- whether and how you would like to be credited.

The maintainers will acknowledge the report, investigate it, coordinate a fix
and disclosure, and keep the reporter informed. Response and release timing
depends on severity, reproducibility, and platform scope; this project does not
promise a timeline it may be unable to meet.

## Security boundaries

- Browser Studio binds to loopback by default. A non-loopback bind requires the
  explicit `--allow-network` or `--lan` option and should only be used on a
  trusted LAN.
- Link sharing is disabled by default. `studio --share-links` can publish only
  the current PNG output after an explicit button press; it does not publish
  source media. `glyph-forge share` exposes exactly the selected file and
  disables the Studio upload endpoint.
- Temporary links are random bearer capabilities. Anyone who receives one and
  can reach the server can read that output until the TTL expires or the
  process stops. Links use unencrypted HTTP and are intended for trusted local
  networks, not the public Internet. Glyph Forge has no hosted relay.
- Webcam and screen access remain subject to browser and operating-system
  permission prompts.
- Installing the public Studio as a PWA does not grant additional media or file
  permissions. Its versioned service worker caches only the static app shell
  and explicitly bypasses `/api/` and `/s/` responses, so private shares and
  live server capabilities do not enter offline storage.
- `live url` contacts the supplied URL and services used by yt-dlp. Treat URLs
  as network operations and use a current yt-dlp release.
- `live launch` executes the exact command supplied by the local user. It does
  not sandbox that application; Xvfb only gives it an isolated display.
- Host-desktop capture remains view-only. Keyboard and pointer forwarding is
  separately opt-in for isolated X11 targets, refuses same-display terminal
  injection, and provides Ctrl+] as an emergency release chord.
- Third-party plugins execute in-process with the user's permissions and are
  not sandboxed. Metadata listing does not import them, but selecting a plugin,
  `plugins inspect`, and `plugins --probe` do. Install only trusted packages;
  set `GLYPH_FORGE_DISABLE_PLUGINS=1` to disable entry-point discovery.
- FFmpeg, OpenCV, yt-dlp, PyVirtualDisplay, and capture backends are optional
  third-party components with their own security policies and update cycles.
- Tagged release assets include a `SHA256SUMS` manifest and GitHub artifact
  attestations signed with an ephemeral Sigstore certificate. Verify both when
  downloading outside a trusted package manager. This provenance is distinct
  from Windows Authenticode signing or macOS notarization; those require
  maintainer-owned platform certificates and are not claimed by the project.

## Defensive design

Glyph Forge keeps optional dependencies lazy, validates local server binds,
sends restrictive browser security headers, and avoids shell-based child-process
launch. Browser publishing uses same-origin and per-session CSRF checks, a PNG
signature check, bounded memory and item counts, and expiring capability tokens.
The HTTP server also bounds simultaneous request threads.
File links are path-resolved, limited to regular files, and revoked if the file
identity, size, or modification time changes. Live buffers are bounded, and
video output uses a temporary destination before atomic replacement. CI checks
formatting, linting, typing, tests, package metadata, optional dependency
installation, installed wheel resources, and Chromium/Firefox/WebKit desktop
and touch behavior. CodeQL scans both Python and browser JavaScript, while
Dependabot monitors Python, npm, and GitHub Actions dependencies.

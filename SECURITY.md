# Security policy

## Supported versions

Security fixes target the current `0.2.x` line and `main`. Older releases may
receive a fix when practical but are not actively supported.

## Report a vulnerability

Do not open a public issue for a vulnerability that could put users at risk.
Send a private report to <lloyd.handyside@neuroforge.io> or
<ace1928@gmail.com> with:

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
  explicit `--allow-network` option and should only be used on a trusted LAN.
- Studio does not upload source media. Browser Web Share hands an export to an
  app selected by the user; style links contain settings, not media.
- Webcam and screen access remain subject to browser and operating-system
  permission prompts.
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

## Defensive design

Glyph Forge keeps optional dependencies lazy, validates local server binds,
sends restrictive browser security headers, avoids shell-based child-process
launch, uses bounded live buffers, and writes video output through a temporary
destination before atomic replacement. CI checks formatting, linting, typing,
tests, package metadata, optional dependency installation, and installed wheel
resources.

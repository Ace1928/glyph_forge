# Changelog

All notable changes are recorded here. The format follows Keep a Changelog and
the project uses semantic versioning.

## [Unreleased]

### Planned

- Native capture adapters and optional compiled rendering hot paths
- A stable extension SDK for third-party sources, renderers, and exporters
- Signed standalone installers and an opt-in hosted sharing service

### Added

- Explicit isolated-desktop keyboard and pointer control through a typed input
  protocol, lazy pynput adapter, UTF-8/SGR terminal parser, viewport coordinate
  mapping, and Ctrl+] emergency release

### Security

- Input forwarding is opt-in, independent from capture permission, and refuses
  unsafe same-display terminal injection that could create an event loop
- Isolated capture, child processes, and input routing receive their target X11
  display explicitly instead of mutating the process-global `DISPLAY`

## [0.2.0] - 2026-08-09

### Added

- One unified `glyph-forge` CLI for image, text, video, live media, diagnostics,
  interface launch, demos, and benchmarks
- Hardware-adaptive eco, balanced, and workstation runtime profiles
- Vectorized full-colour glyph-video export with FFmpeg streaming, audio muxing,
  subclips, quality controls, progress reporting, and atomic destinations
- Bounded-latency webcam, video, and screen capture using a newest-frame slot
- Glyph, directional edge, Braille 2×4, true-colour half-block, quadrant, and
  real-text SVG renderers
- Sobel, Prewitt, Scharr, Laplacian, and Canny-style edge detection
- Optional yt-dlp URL playback without downloading media to disk
- Optional isolated X11 app launch through PyVirtualDisplay and Xvfb
- Private browser Studio with drag/drop, video, webcam, screen capture, WebGL2
  and Canvas rendering, PNG/SVG/TXT exports, Web Share, and style links
- Rebuilt Textual TUI with media browsing, previews, live sources, saving,
  diagnostics, and Studio handoff
- Deterministic built-in demo and renderer benchmark commands
- Portable bundled Eidos profile defaults with user-writable overrides

### Changed

- Integrated the standalone video script and useful legacy prototype features
  into tested package modules and CLI commands
- Replaced eager or unbounded video helpers with lazy iterators while retaining
  list-returning compatibility functions
- Reduced `imagize`, `bannerize`, and virtual-display compatibility surfaces to
  thin adapters over maintained implementations
- Consolidated development history onto `main`, made it the GitHub default, and
  removed every redundant local and remote branch
- Replaced overlapping Black/isort/flake8 workflows with one Ruff configuration,
  strict mypy checks, a cross-platform test matrix, and installed-wheel tests
- Simplified packaging metadata, optional extras, examples, and documentation

### Fixed

- Public `glyph_forge.image_to_glyph` now resolves to the callable helper rather
  than the service module
- Legacy `image_to_Glyph` imports now use an in-memory alias instead of a
  case-colliding compatibility file on Windows and macOS
- `doctor` launches FFmpeg tools instead of trusting executable names alone
- Configuration discovery no longer creates user directories during import
- ASCII borders use the correct corner, vertical, and horizontal characters
- Standalone compatibility help and legacy short-option translation
- The former duplicate `-h` option and non-existent image optimization call

### Security

- Studio binds to loopback by default and requires `--allow-network` for a
  non-loopback address
- Studio sends restrictive CSP, no-cache, no-sniff, frame, and referrer headers
- Network and virtual-display dependencies remain lazy and opt-in

### Repository

- Removed obsolete generated art, editor settings, copied tutorials, templates,
  prompt files, duplicate scripts, stale status files, and deprecated setup files
- Retained compatibility through small tested adapters and Git history rather
  than parallel implementations

## [0.1.0] - 2024-11-15

### Added

- Image-to-glyph conversion and density character sets
- FIGlet text banners and style presets
- Text, ANSI, HTML, and SVG output renderers
- Initial `imagize`, `glyphfy`, and `bannerize` commands
- Public API, configuration, service, transformer, and utility modules
- Initial tests, package metadata, and contribution documentation

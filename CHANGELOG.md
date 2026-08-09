# Changelog

All notable changes are recorded here. The format follows Keep a Changelog and
the project uses semantic versioning.

## [Unreleased]

### Planned

- Native capture adapters and optional compiled rendering hot paths
- Signed standalone installers and an opt-in hosted sharing service

### Added

- Browser Studio parity for density glyph, directional edge, Braille 2×4,
  quadrant 2×2, and true-colour half-block rendering, plus local text sources
  and a fullscreen output surface
- One-click full-file and manually controlled live Studio recording, combining
  the rendered canvas with available source audio on one media timeline and
  selecting a supported WebM or MP4 encoder at runtime
- Cross-platform PyInstaller one-directory bundles, deterministic ZIP/tar
  archives, double-build reproducibility checks, portable smoke tests, SHA-256
  manifests, and Sigstore-signed GitHub artifact provenance for tagged releases
- Fast isolated installation and one-off launch guidance for `uv`, alongside
  pipx, wheels, and no-Python release archives
- Opt-in ephemeral capability links for Studio PNG snapshots and exact local
  files, including the `glyph-forge share` command, trusted-LAN address
  discovery, configurable expiry, and browser copy-to-clipboard workflow
- HTTP range and HEAD support for seekable image, audio, and video links, with
  IPv4/IPv6 URL generation and an explicit advertised-host override
- Explicit isolated-desktop keyboard and pointer control through a typed input
  protocol, lazy pynput adapter, UTF-8/SGR terminal parser, viewport coordinate
  mapping, and Ctrl+] emergency release
- Adaptive terminal presentation with `auto`, `delta`, and `full` redraw modes,
  exact output-byte accounting, resize fallback, and public presenter contracts
- Aspect-preserving live viewport fitting across every glyph mode, with
  immediate terminal-resize adaptation and an explicit `--no-fit` override
- Versioned plugin API v1 for third-party sources, renderers, transforms, and
  exporters, with metadata-only discovery, explicit probing, per-plugin failure
  isolation, in-process registration, and CLI diagnostics
- `live source` and top-level `stream` commands that route paths, URLs, devices,
  and plugin sources through one bounded-latency pipeline
- Complete machine-readable video export metrics covering duration, throughput,
  worker count, source/output/raw sizes, glyph rate, and real-time factor

### Performance

- Compact 2D GPU glyph atlases, reusable video textures, character-correct
  aspect ratios, and decoded-frame callbacks for high-detail Studio rendering
  without refresh-rate duplicate work
- File-backed sharing streams large outputs in place through the platform
  `sendfile` path when available, with a bounded portable fallback and no
  media-sized memory copy
- Vectorized half-block palette conversion and run-length terminal emission,
  removing per-cell NumPy allocations and redundant ANSI256 colour sequences
- Changed-row terminal updates which automatically fall back to complete frames
  when cursor-addressed output would be larger
- Per-frame viewport constraints are applied before image resampling, avoiding
  wasted rendering work for pixels that cannot fit on the terminal surface
- Offline video frames render through a bounded hardware-adaptive worker pool
  while retaining exact source order, deterministic output, and audio sync

### Fixed

- Redirected and frozen Windows commands switch their existing text streams to
  UTF-8 before emitting Braille, block, or international glyphs

### Security

- Link sharing is disabled by default and bounded by random capability tokens,
  TTL, item count, and memory; browser publication requires same-origin CSRF
  validation and a real PNG signature, while file links reject changed files
- Input forwarding is opt-in, independent from capture permission, and refuses
  unsafe same-display terminal injection that could create an event loop
- Isolated capture, child processes, and input routing receive their target X11
  display explicitly instead of mutating the process-global `DISPLAY`

### Repository

- Release automation validates the full suite, builds Windows/macOS/Linux apps,
  and exercises the frozen CLI, renderer, and packaged Studio before publishing
- Package validation now rejects stale case-colliding legacy modules in built
  wheels

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

- Android CLI sessions relaunch once without foreign `LD_LIBRARY_PATH`
  overrides before loading native media modules, child media tools receive the
  same clean environment, and video exports stop at the shorter mapped stream
  instead of retaining a trailing audio-only tail

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

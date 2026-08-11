# Changelog

All notable changes are recorded here. The format follows Keep a Changelog and
the project uses semantic versioning.

## [Unreleased]

### Planned

- Native capture adapters and optional compiled rendering hot paths
- Signed standalone installers and an opt-in hosted sharing service
- One-time PyPI owner setup for the unambiguous `glyphforge` distribution;
  versioned GitHub wheels and portable archives remain the canonical release
  path until then

## [0.4.0] - 2026-08-11

### Added

- Public render contract v1: immutable, JSON-serializable `RenderRequest`,
  structured `RenderArtifact`, phase-level `RenderMetrics`, normalized format,
  fit and alignment enums, and typed contract/source/render/export errors
- One canonical `render_image` path for filesystem, Pillow, and NumPy sources,
  including EXIF orientation, transparent-image compositing, bounded decoding,
  safe HTML, ANSI256/truecolor, real-text SVG, and PNG encoding
- User-friendly `glyph-forge image --size WIDTHxHEIGHT`, independent cell and
  pixel geometry, contain/cover/stretch behavior, nine anchors, and exact
  geometry reporting in CLI and TUI saves
- All five still modes in the TUI, with tone controls, exact graphical export,
  HTML/ANSI handling, responsive scrollable forms, and background rendering
- Schema-versioned, layered, platform-native configuration with read-only
  system defaults, atomic user persistence, session overrides, legacy JSON
  migration, validation, thread safety, and rollback after failed writes
- Shared Python/JavaScript golden fixtures covering density mapping, integer
  RGB luma, tone rounding, truecolor, Braille and quadrant bit order,
  half-block cells, and Sobel edge direction/weight
- Full public still-pipeline benchmarking through `benchmark --pipeline`, with
  an eco-profile latency regression budget alongside kernel measurements
- Rendering-contract and 0.4 migration guides, including the compatibility
  schedule through 1.0

### Changed

- CLI, TUI, `GlyphForgeAPI`, the public helper, legacy still adapter, and the
  frame-list video compatibility service now delegate to the maintained
  renderer instead of selecting parallel still implementations
- Named density, special, and language character sets use one thread-safe,
  deterministic resolver; likely preset typos now include suggestions and
  custom lowercase-only sets use the explicit `literal:` prefix
- `.html` and `.ansi` destinations select their format by suffix, while PNG
  and SVG color comes from explicit foreground/background controls
- Browser Canvas exports use native-compatible tone rounding, integer luma,
  and Sobel direction/weight; the installed Studio shell advances to cache v3
- Text, artifacts, API saves, configuration/profile data, GlyphCode restores,
  demo outputs, and compatibility saves share one crash-safe, fsynced atomic
  persistence primitive; pre-0.4 Windows/macOS configuration and profile paths
  migrate without deleting the legacy recovery copy
- The image CLI was split into focused request/catalog adapters and now stays
  below the repository's strict complexity ceiling

### Deprecated

- `ImageGlyphConverter` in favor of `RenderRequest` plus `render_image`
- `GlyphForgeAPI.image_to_Glyph` in favor of `image_to_glyph`
- Top-level `get_config(profile)` in favor of `get_profile_config(profile)`;
  persistent settings use `get_settings()`
- These compatibility APIs remain tested throughout 0.x and are scheduled for
  removal at 1.0

### Fixed

- Malformed serialized request types now produce stable contract errors rather
  than leaking `TypeError` or `AttributeError`
- Failed configuration replacements restore both persistent and runtime state,
  so memory cannot claim a value that was not committed to disk
- Compact TUI terminals can reach every image, text, and live control through
  a real scroll container
- HTML output inferred from `.html` no longer silently writes plain text when
  the color flag is omitted

### Performance

- The canonical path downsamples before cached tone mapping and uses the
  maintained vectorized frame kernels for every still mode
- Browser/native edge conformance uses a reduced-grid Sobel pass, keeping work
  proportional to visible glyphs while aligning exported text semantics

### Security

- Still sources are bounded to 100 megapixels, request dimensions and numeric
  types are validated, HTML content/colors are escaped, and partial output
  files are never published

## [0.3.1] - 2026-08-11

### Added

- Exact, independent output dimensions throughout Studio: adaptive, source,
  720p, 1080p, 1440p, 4K, 8K, and custom width/height presets, with optional
  aspect locking and device-safe GPU/canvas bounds
- First-class CLI PNG and real-text SVG still exports inferred from the output
  suffix, with `--output-width` and `--output-height` kept separate from glyph
  columns and rows, automatic one-axis aspect preservation, and configurable
  foreground/background colours
- Public lazy `render_text_png` and `render_text_svg` APIs for exact-size
  application exports
- Browser acceptance coverage proving exact custom canvas/SVG dimensions stay
  unchanged when glyph density changes

### Changed

- Brighter, clearer visual defaults (`1.12` brightness and `1.08` contrast)
  are now consistent across grayscale and colour stills, live renderers,
  Studio, video exports, configuration, and style links
- Studio SVG exports now use the selected pixel dimensions as their exact
  viewBox and stretch real text rows across that vector surface
- Native video pixel dimensions no longer need to divide evenly by glyph rows
  and columns; divisible configurations retain the direct fast path and custom
  fractional cells receive a high-quality exact-size fit
- The Studio offline shell is versioned to v2 so installed apps receive the
  new sizing controls and visual defaults immediately

### Fixed

- Colour image conversion now applies brightness and contrast instead of
  silently bypassing both controls
- API converter overrides preserve configured tone values and correctly accept
  an explicit zero brightness or contrast value

### Performance

- Native still, live, and video tone adjustment shares one cached 256-entry
  NumPy lookup table and runs after spatial downsampling, avoiding full-source
  per-pixel tone math in latency-sensitive paths

### Security

- Future release assets are assembled as drafts before publication so GitHub
  can lock their tag and files through enforced immutable releases

## [0.3.0] - 2026-08-11

### Added

- Installable cross-device Studio PWA with branded icons, standalone display,
  safe-area-aware mobile layout, offline app shell, file launch integration,
  project-relative hosting, and platform-specific install guidance
- Automated Chromium, Firefox, and WebKit acceptance checks with Pixel and
  iPhone touch profiles, accessibility scanning, responsive geometry, offline
  operation, manifest validation, and a browser source-size budget
- CodeQL scanning for Python and browser JavaScript, Dependabot security monitoring,
  structured bug/feature/performance reports, and a production pull-request
  checklist
- Portable glyph codes: encode any image, banner, or animated GIF into one
  printable ASCII string (`glyph:v1:…`) that regenerates the original artwork
  with no hosting or server — `glyph-forge link code|banner|decode`, lossless
  byte-exact image round trips, banner style retention, and per-frame GIF
  timing; the browser Studio and GitHub Pages playground accept pasted codes
- Self-contained `glyph-forge demo` showcase: an entertaining, self-sufficient
  tour of the whole toolkit, with meme templates and popular video thumbnails
  downloaded at runtime (deterministic offline stand-ins when the network is
  unavailable), every banner font, character set, text style, and render mode,
  and per-scene artifacts saved with `--output-dir`
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

### Changed

- The installable Python distribution is now the unambiguous `glyphforge`
  project while the product, `glyph-forge` executable, and `glyph_forge` import
  stay unchanged; documentation warns against the unrelated hyphenated PyPI
  package
- Every CLI command, option, and argument now carries an explicit help
  description, so `--help` fully explains each workflow without reading docs
- The repeated live-media options (mode, colour, width, charset, edges, timing,
  performance) are defined once through shared builders so their help and
  defaults stay consistent across `live source`, `camera`, `screen`, `video`,
  `url`, `launch`, and their top-level aliases
- The `examples` directory is now a regular package so the runnable plugin
  example imports even when an unrelated `examples` module shadows it earlier
  on `sys.path`

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

- Studio file input naming and coarse-pointer text controls now meet the same
  accessible-name and minimum touch-target gates as visible controls
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

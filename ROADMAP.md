# Glyph Forge roadmap

Glyph Forge is a local-first, portable glyph rendering engine with friendly
CLI, TUI, browser, API, live-media, desktop-viewing, video, and sharing
surfaces. Work is sequenced by user-visible outcomes and measurable acceptance
gates rather than by feature count.

## Current foundation

| Track | Current state |
|---|---|
| Portable core | Python 3.10–3.14, lazy optional dependencies, hardware profiles, wheels and portable archives |
| User surfaces | Unified CLI, scrollable Textual TUI, installable touch-first PWA, typed Python API |
| Rendering | Glyph, Sobel/Prewitt/Scharr/Laplacian/Canny edge, Braille, half-block, quadrant, ANSI/HTML/PNG/real-text SVG |
| Media | Files, camera, screen, URLs, bounded-latency terminal playback, streamed FFmpeg video with audio |
| Desktop | Host viewer plus opt-in interactive isolated X11 application display |
| Sharing | Lossless glyph codes, Web Share, bounded local/LAN seekable links |
| Extensions | Plugin API v1 for sources, renderers, transforms, and exporters |
| Assurance | Cross-OS Python, browser/touch/accessibility, package, bundle, CodeQL, coverage, and performance gates |

## Phase 1 — one trustworthy still renderer (0.4)

Status: complete.

- Versioned `RenderRequest` contract and structured `RenderArtifact` result.
- One canonical still pipeline behind CLI, TUI, API, and compatibility calls.
- Independent character density and exact PNG/SVG output size, with contain,
  cover, stretch, and nine alignment anchors.
- Brighter shared tone defaults, alpha compositing, EXIF orientation, bounded
  inputs, strict types, and actionable character-set typo recovery.
- Named density, special, and language alphabets with explicit `literal:`
  custom syntax.
- Typed load/render/export errors and atomic artifact/config/text persistence.
- Platform-native versioned configuration with transactional rollback.
- Shared Python/JavaScript golden corpus covering all five modes, RGB luma,
  tone rounding, truecolor, bit order, and Sobel direction/weight.
- Complete-pipeline benchmark gate in addition to kernel benchmarks.
- Compatibility deprecations scheduled for 1.0 and documented migration.

Acceptance evidence lives in `tests/test_rendering_contract.py`,
`tests/test_image_exports.py`, `tests/test_tui.py`, `tests/test_config.py`,
`tests/test_persistence.py`, `tests/test_benchmark.py`, and
`tests/fixtures/render-contract-v1.json`.

## Phase 2 — project workflow and complete media parity

Goal: make long creative sessions as effortless and deterministic as one still.

- Introduce a versioned project/preset document around `RenderRequest`, with
  recent files, autosave, undo/redo, non-destructive variants, and portable
  relative asset references.
- Give CLI, TUI, and Studio equivalent preset import/export and batch queues.
- Route native video and live sessions through a temporal render contract that
  shares every still mode, palette, fit, alphabet, and tone definition.
- Add all-mode high-resolution offline video export, explicit quality/speed
  presets, resumable jobs, cancellation, frame/audio progress, and machine-
  readable benchmark reports.
- Add deterministic audio-clock synchronization tests for variable frame rate,
  subclips, dropped live frames, and long exports.
- Add side-by-side source/output, zoom/pan, histogram, and before/after tone
  tools without duplicating renderer state.

Exit gates: project round trips across all interfaces; pixel/glyph parity for
every still and temporal mode; bounded cancellation latency; long-run A/V drift
below one video frame; recovery tests for interrupted jobs.

## Phase 3 — zero-copy live and interactive desktop engine

Goal: a safe, buttery glyph desktop on modest hardware that scales to rendering
workstations.

- Add native capture adapters behind the existing source protocol: Windows
  Graphics Capture/Desktop Duplication, macOS ScreenCaptureKit, Linux X11
  XShm/XComposite, and Wayland PipeWire portal.
- Negotiate damage regions, pixel formats, GPU handles, refresh rate, and color
  space to avoid full-frame copies where a platform supports them.
- Add optional compiled SIMD kernels only after benchmark evidence, retaining
  NumPy/Pillow as the fully supported fallback.
- Evaluate compute/WebGPU/CUDA/Metal/Vulkan backends through one capability
  contract, with identical golden output and automatic safe fallback.
- Complete separately permissioned input adapters: Windows SendInput, macOS
  Accessibility, X11 XTest, and user-mediated Wayland support. Preserve the
  same-display feedback-loop rejection and emergency-release chord.
- Add latency telemetry from capture timestamp to present, adaptive quality
  control, frame pacing, and terminal/GPU damage-only presentation.

Exit gates: p50/p95 motion-to-present budgets per supported backend; no
unbounded queues; deterministic fallback under device loss; permission and
emergency-stop tests; multi-hour leak/stability runs.

## Phase 4 — first-class desktop distribution

Goal: install and launch without needing to understand Python.

- Package a polished desktop shell around the web renderer and native engine,
  with file associations, drag/drop, native menus, keyboard/touch parity,
  background job notifications, and safe update checks.
- Produce signed Windows installers, notarized macOS universal applications,
  Linux AppImage/Flatpak packages, and retained portable archives.
- Add architecture coverage for x86-64 and ARM64 where upstream dependencies
  permit it, plus Windows on ARM and Apple Silicon smoke hardware.
- Establish reproducible build manifests, SBOMs, artifact signatures,
  rollback-safe updates, and release-channel policy.

Exit gates: clean-machine install/use/uninstall checks; OS signing and
notarization verification; offline launch; upgrade/downgrade project
compatibility; accessibility review on each desktop OS.

## Phase 5 — sharing, discovery, and launch assets

Goal: make finished work delightful to publish while keeping local-first
privacy obvious.

- Add opt-in hosted relay/storage as a separate service, with accounts not
  required for local use, explicit expiry, deletion, quotas, abuse controls,
  encryption in transit, and transparent privacy boundaries.
- Create small share pages with responsive playback, embeds, thumbnails,
  attribution, remixable style presets, and downloadable originals when the
  creator permits them.
- Add curated templates, visual preset packs, onboarding challenges, and a
  reproducible showcase generator for launch clips and benchmark galleries.
- Produce flagship audio-synchronized demos only from user-provided or properly
  licensed media, with automatic source/output size, duration, render time,
  FPS, real-time factor, and quality summaries.

Exit gates: revocation and expiry tests; content-security and abuse review;
portable export when the service is unavailable; explicit rights/provenance
records for published showcase media.

## Phase 6 — 1.0 stability and ecosystem

Goal: a durable commercial-production-quality platform rather than a permanent
prototype.

- Freeze render contract v1 and plugin API v1 after an external beta period.
- Remove scheduled compatibility APIs only at 1.0, with an automated migration
  checker and a complete final 0.x bridge release.
- Publish support, security-response, compatibility, benchmark, and release
  service-level policies that the maintainers can actually sustain.
- Add opt-in privacy-preserving crash diagnostics, never source media or
  artwork, with a fully functional no-telemetry default.
- Build reference plugins, conformance kits, API docs, architecture decision
  records, contributor fixtures, and long-term maintenance ownership.

Exit gates: no undocumented public behavior; two-release project compatibility;
supported-platform matrix green; zero critical security findings; measured
performance baselines; complete install-to-share user documentation.

## Invariants for every phase

- `main` is the only development branch; releases are immutable tags.
- Imports remain side-effect-free and optional dependencies remain lazy.
- Source media stays local unless the user explicitly publishes it.
- Capture permission never implies input permission.
- Queues, memory, dimensions, uploads, and subprocesses remain bounded.
- Optimization requires reproducible before/after evidence and output parity.
- Compatibility is a thin tested adapter, never a second implementation.
- Formatting, linting, strict typing, tests, coverage, package, bundle, browser,
  security, and platform gates must be green before a release.

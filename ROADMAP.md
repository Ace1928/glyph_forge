# Glyph Forge roadmap

Glyph Forge is becoming a portable glyph rendering engine while keeping one
tested pipeline behind the CLI, TUI, browser Studio, and Python API.

## Product tracks

| Track | User outcome | Status |
|---|---|---|
| Portable core | Install, import, diagnose, and run on small or large systems | Wheels and portable app archives complete |
| Unified experiences | Friendly CLI, TUI, multimode recording Studio, and compatibility launchers | Complete in 0.2 |
| Installable web app | Offline-capable, touch-first PWA across Chromium, Firefox, WebKit, Android, and Apple devices | Complete in 0.3 |
| Live media | Stream files, cameras, screens, and URLs with bounded latency | Complete in 0.2 |
| Desktop viewing | Render a desktop or isolated X11 app through glyph modes | Viewer complete in 0.2 |
| Desktop control | Opt-in focused keyboard and pointer forwarding | Isolated X11 complete; native host adapters active |
| Sharing | Exports, style links, and ephemeral seekable LAN links | Local/LAN complete; hosted relay optional later |
| Native acceleration | Browser GPU atlas and bounded parallel NumPy video path now; compiled kernels later | Active |
| Extension SDK | Third-party source, renderer, transform, and exporter plugins | API v1 complete in 0.3 |
| Production assurance | Native OS, browser, touch, accessibility, package, dependency, and security gates | Active; automated baseline complete |

## Fidelity and performance

The maintained modes are `glyph`, `edge`, `braille`, `half-block`, and
`quadrant`, across the terminal engine and Studio GPU atlas, plus scalable SVG
still export and browser-native audio-synced recording. The capture
pipeline keeps bounded buffers and favors latency over rendering every frame.
Resolution, worker count, sampling, and FPS come from a portable runtime profile
and remain explicitly overridable. Terminal presentation also selects between
changed-row and complete-frame updates from their measured payload sizes. Live
views preserve their aspect ratio inside the current terminal and re-evaluate
the viewport on every frame, so window resizing does not require a restart or
render detail that cannot be displayed.

As of 0.3.1, final PNG/SVG/video pixels are independent from glyph columns and
rows. Studio offers exact dimensions through 8K with aspect locking and
hardware-safe limits, while native video retains a direct fast path for
integral cells. A shared cached tone lookup runs after downsampling, providing
brighter defaults without adding full-frame work to live rendering.

Benchmark automation is available now through `glyph-forge benchmark`. Future
optimization work must use repeatable before/after measurements and retain the
portable NumPy/Pillow fallback.

## Desktop control boundary

Capture and input injection are separate permissions. Input forwarding is
explicitly opt-in and provides Ctrl+] as an immediate escape chord. Pointer
coordinates map through the visible viewport before platform adapters receive
them. Isolated X11 targets are available now. A same-display terminal target is
rejected because injected input can feed back into the focused terminal.

| Platform | Preferred capture direction | Input/permission direction |
|---|---|---|
| Windows | Desktop Duplication or Windows Graphics Capture | SendInput with explicit activation |
| macOS | ScreenCaptureKit | Screen Recording and Accessibility grants |
| Linux X11 | XShm/XComposite | XTest scoped to the selected display |
| Linux Wayland | PipeWire desktop portal | User-mediated portal/compositor support |
| Portable fallback | MSS or Pillow | Viewer only; typed no-op input sink |

## Delivery sequence

1. ~~Complete 0.2 packaging, cross-platform CI, documentation, and release
   checks.~~ Completed.
2. ~~Add an input-routing protocol with a no-op default and platform capability
   reporting.~~ Completed.
3. ~~Implement and test isolated X11 routing first, including coordinate
   mapping, emergency release, and virtual-display interaction.~~ Completed.
4. Add native Windows, macOS, and Wayland capture adapters behind the existing
   source protocol.
5. Profile hot paths and introduce optional compiled kernels only where measured
   gains justify their maintenance and packaging cost. Portable passes removed
   per-cell half-block colour conversion and added bounded, ordered parallel
   video rendering, keeping NumPy fast enough to defer a compiled dependency.
   The Studio reuses GPU source textures and renders video on decoded-frame
   callbacks, avoiding redundant work at display refresh rate.
6. ~~Define versioned plugin contracts and isolated discovery for external
   sources, renderers, transforms, and exporters.~~ Completed with lazy entry
   points, live-pipeline integration, diagnostics, and failure isolation.
7. ~~Add bounded, opt-in, seekable local/LAN links without changing the
   local-first default.~~ Completed.
8. ~~Produce reproducible wheels plus tested Windows, macOS, and Linux portable
   app archives with checksums and signed build provenance.~~ Completed.
   Certificate-backed Windows Authenticode and macOS notarization remain a
   release-operations task; evaluate an opt-in hosted relay separately.

Every milestone must keep imports side-effect-free, optional dependencies lazy,
buffers bounded, compatibility adapters tested, and the cross-platform CI suite
green.

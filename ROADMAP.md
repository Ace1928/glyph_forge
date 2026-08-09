# Glyph Forge roadmap

Glyph Forge is becoming a portable glyph rendering engine while keeping one
tested pipeline behind the CLI, TUI, browser Studio, and Python API.

## Product tracks

| Track | User outcome | Status |
|---|---|---|
| Portable core | Install, import, diagnose, and run on small or large systems | Complete in 0.2 |
| Unified experiences | Friendly CLI, TUI, browser Studio, and compatibility launchers | Complete in 0.2 |
| Live media | Stream files, cameras, screens, and URLs with bounded latency | Complete in 0.2 |
| Desktop viewing | Render a desktop or isolated X11 app through glyph modes | Viewer complete in 0.2 |
| Desktop control | Opt-in focused keyboard and pointer forwarding | Isolated X11 complete; native host adapters active |
| Sharing | Local exports and style links; optional hosted links later | Active |
| Native acceleration | Browser GPU atlas now; compiled/native hot paths later | Active |
| Extension SDK | Third-party source, renderer, transform, and exporter plugins | Planned for 0.4 |

## Fidelity and performance

The maintained modes are `glyph`, `edge`, `braille`, `half-block`, and
`quadrant`, plus scalable SVG still export and the Studio GPU atlas. The capture
pipeline keeps bounded buffers and favors latency over rendering every frame.
Resolution, worker count, sampling, and FPS come from a portable runtime profile
and remain explicitly overridable. Terminal presentation also selects between
changed-row and complete-frame updates from their measured payload sizes. Live
views preserve their aspect ratio inside the current terminal and re-evaluate
the viewport on every frame, so window resizing does not require a restart or
render detail that cannot be displayed.

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

1. Complete 0.2 packaging, cross-platform CI, documentation, and release checks.
2. ~~Add an input-routing protocol with a no-op default and platform capability
   reporting.~~ Completed.
3. ~~Implement and test isolated X11 routing first, including coordinate
   mapping, emergency release, and virtual-display interaction.~~ Completed.
4. Add native Windows, macOS, and Wayland capture adapters behind the existing
   source protocol.
5. Profile hot paths and introduce optional compiled kernels only where measured
   gains justify their maintenance and packaging cost. The first pass removed
   per-cell half-block colour conversion and made the portable NumPy path fast
   enough to defer a compiled dependency.
6. Define versioned plugin contracts and isolated discovery for external
   sources, renderers, and exporters.
7. Produce reproducible signed wheels, standalone installers, and opt-in hosted
   sharing without changing the local-first default.

Every milestone must keep imports side-effect-free, optional dependencies lazy,
buffers bounded, compatibility adapters tested, and the cross-platform CI suite
green.

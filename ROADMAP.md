# Glyph Forge product roadmap

Glyph Forge is evolving from a media converter into a portable glyph rendering
engine. The work is deliberately layered so every interface uses the same
tested conversion, capture, rendering, sharing, and capability APIs.

## Product tracks

| Track | User outcome | Status |
|---|---|---|
| Portable core | Install, import, diagnose, and run on modest or powerful hardware | Complete in 0.2 |
| Unified experiences | Friendly CLI, full-screen TUI, and local browser studio | Complete in 0.2 |
| Live media | Stream files, cameras, and screens without loading whole videos | Complete in 0.2 |
| Desktop mirror | View and optionally control a desktop through glyph rendering | Planned for 0.3 |
| Sharing | Local exports and style links now; opt-in hosted links later | Active in 0.2 |
| Extension SDK | Third-party capture, transform, renderer, and exporter plugins | Planned for 0.4 |
| Native acceleration | Browser GPU atlas now; optional compiled/native hot paths later | Active in 0.2 |

## Live rendering architecture

```text
native capture -> latest-frame buffer -> transform -> renderer -> presentation
      |                                      |             |
 Windows / macOS / X11 / Wayland       glyph / braille   terminal
 webcam / video / screen               blocks / color    browser / GPU window
                                              |
                                      SVG still export
```

The pipeline has bounded buffers and drops stale frames under load. Latency is
more important than rendering every captured frame. Resolution, sampling,
worker count, and target frame rate come from the portable runtime profile and
remain explicitly overridable.

### Fidelity modes

- `glyph`: density-mapped characters; the most visibly typographic mode.
- `braille`: 2×4 binary subcells per Unicode character for eight spatial
  samples per terminal cell.
- `half-block`: independent true-color upper and lower pixels per character.
- `quadrant`: 2×2 subcell geometry where terminal/font support is reliable.
- `atlas`: a GPU-rendered character atlas for the highest-resolution live GUI.
- `svg`: scalable, lossless still output with real text glyphs.

Live browser/native views use a GPU-backed canvas or texture atlas. SVG is an
export format, not the per-frame live DOM, because thousands of text nodes per
frame would add avoidable latency.

## Desktop mirror and interaction

Desktop mode is a viewer/control surface, not a security bypass. Capture and
input injection are separate capabilities and input forwarding is always an
explicit opt-in. The first implementation will map pointer coordinates through
the rendered viewport and forward keyboard/pointer events only while the view
is focused.

| Platform | Preferred capture | Permission considerations |
|---|---|---|
| Windows | Desktop Duplication / Windows Graphics Capture | User/session access |
| macOS | ScreenCaptureKit | Screen Recording and Accessibility prompts |
| Linux X11 | XShm/XComposite | Display/session access |
| Linux Wayland | PipeWire via desktop portal | User-mediated portal grant |
| Portable fallback | MSS screenshot capture | Backend-dependent |

Platform backends implement one capture protocol. Unsupported capture or input
features appear in `glyph-forge doctor` with an actionable installation or
permission hint; they never prevent image/text workflows from starting.

## Delivery sequence

1. Side-effect-free imports, adaptive runtime profiles, unified commands, and
   dependable package metadata.
2. Vectorized renderers plus streaming video export. **Complete in 0.2.**
3. Webcam capture with the same bounded latest-frame pipeline. **Complete in
   0.2.**
4. Screen capture and a terminal desktop mirror with bounded-latency
   scheduling. **Viewer complete in 0.2; opt-in input routing remains.**
5. Browser Studio with live preview, drag-and-drop files, downloads, style
   links, and browser-native sharing. **Complete in 0.2.**
6. TUI parity, runtime diagnostics, native capture adapters, and benchmarks.
   **TUI and diagnostics complete; native adapters and benchmark automation
   remain.**
7. Desktop input routing, compiled hot paths, plugin SDK, signed releases, and
   standalone installers.

Each milestone must keep the core test suite green, add focused tests and
benchmarks for new hot paths, preserve compatibility aliases, and avoid making
optional heavyweight dependencies part of basic startup.

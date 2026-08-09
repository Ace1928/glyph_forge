# Glyph Forge

[![CI](https://github.com/Ace1928/glyph_forge/actions/workflows/ci.yml/badge.svg)](https://github.com/Ace1928/glyph_forge/actions/workflows/ci.yml)
[![Python 3.10–3.14](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue)](https://www.python.org/)
[![MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Glyph Forge turns images, text, videos, webcams, screens, and supported media
URLs into character art. The same rendering engine is available through a
direct CLI, a full-screen terminal UI, a private browser Studio, and a Python
API.

It is designed to start quickly on modest hardware and scale its defaults on a
workstation. Live sessions keep one newest-frame slot instead of building a
queue, so a slow renderer drops stale frames rather than accumulating latency.

## Install

Glyph Forge supports Python 3.10 through 3.14 on Windows, macOS, Linux, and
other Python environments supported by its core dependencies.

```bash
# Small core: images, text, browser Studio, demo, and diagnostics
python -m pip install glyph-forge

# Everything, including the TUI and live-media backends
python -m pip install "glyph-forge[all]"
```

For an isolated command-line installation, use pipx:

```bash
pipx install "glyph-forge[all]"
```

Install only what you use:

| Extra | Adds |
|---|---|
| `tui` | Full-screen Textual interface |
| `media` | OpenCV video/webcam capture and fast MSS screen capture |
| `network` | yt-dlp URL resolution for `live url` |
| `virtual` | PyVirtualDisplay support for isolated X11 applications |
| `control` | Explicit keyboard and pointer forwarding through pynput |
| `all` | Every optional interface and backend |
| `dev` | Tests, Ruff, mypy, build, and release checks |

Full-colour video export also needs `ffmpeg` and `ffprobe` on `PATH`. The
isolated app display needs Xvfb and is available on Linux/FreeBSD X11 systems.
Run `glyph-forge doctor` for exact capability and installation guidance.

## Start anywhere

```bash
# Let Glyph Forge choose the friendliest installed interface
glyph-forge launch

# Or choose explicitly
glyph-forge launch gui
glyph-forge launch tui
glyph-forge launch cli

# Immediate no-input showcase and hardware report
glyph-forge demo
glyph-forge doctor
```

`glyph-forge` with no arguments prints a compact quick start. Every command has
focused help, for example `glyph-forge image --help`.

## Images and text

```bash
# Preview in the terminal
glyph-forge image photo.jpg --width 90

# Direction-aware line glyphs using the built-in NumPy edge engine
glyph-forge image photo.jpg --mode edge --edge-algorithm scharr
glyph-forge image photo.jpg --mode edge --edge-algorithm canny --edge-threshold 64

# Subpixel and colour-oriented modes
glyph-forge image photo.jpg --mode braille --output photo.txt
glyph-forge image photo.jpg --mode half-block --color ansi

# Browse before choosing
glyph-forge image --list-charsets
glyph-forge image --preview-charset detailed --sample photo.jpg

# FIGlet banners, fonts, styles, preview, and saving
glyph-forge text "GLYPH FORGE" --font slant --style boxed --preview
glyph-forge text --list-fonts
glyph-forge text --list-styles
glyph-forge text "SHARE THIS" --output banner.txt
```

Image outputs support plain text, ANSI colour, and HTML. Still frames can also
be rendered as real-text SVG through the Python API, retaining sharp glyphs at
arbitrary zoom levels.

## Rendering modes

| Mode | Detail per terminal cell | Best use |
|---|---:|---|
| `glyph` | one density-mapped character | clearly visible typography |
| `edge` | one directional line character | structure, UI, and line art |
| `braille` | 2×4 binary subcells | highest terminal spatial detail |
| `half-block` | two independent colour samples | colour-rich terminals |
| `quadrant` | 2×2 binary subcells | compact geometric detail |

Edge mode includes Sobel, Prewitt, Scharr, Laplacian, and Canny-style
detectors. All detectors operate on the reduced output grid with vectorized
NumPy operations, keeping their cost proportional to visible cells.

## Video, webcam, screen, and URLs

Install `glyph-forge[media]` for the primary live backends.

```bash
# Memory-bounded terminal playback
glyph-forge live video clip.mp4 --mode braille
glyph-forge webcam 0 --mode braille --fps 30
glyph-forge desktop 1 --mode half-block --color truecolor
glyph-forge desktop 1 --redraw auto

# One resolver for built-in paths/URLs/devices and plugin sources
glyph-forge stream "plugin:visualizers/synth:440hz" --mode braille

# Supported sites are resolved without saving the media to disk
glyph-forge live url "https://example.com/video-page" --mode edge

# Encode a full-colour glyph video and preserve source audio
glyph-forge video clip.mp4
glyph-forge video clip.mp4 output.mp4 --performance workstation --crf 16
```

The exporter streams OpenCV frames directly through a vectorized glyph atlas
to FFmpeg. It does not create a temporary image sequence, and the destination
is replaced only after encoding succeeds.

Live terminal views default to adaptive redraws. Glyph Forge compares the
actual UTF-8 payload for a complete frame with cursor-addressed changed rows,
uses whichever is smaller, and sends nothing when both the surface and footer
are unchanged. `--redraw delta` forces row updates and `--redraw full` provides
a compatibility fallback; redirected output always stays full-frame and
line-oriented.

Interactive views also fit every frame inside the current terminal viewport,
preserve the source aspect ratio, reserve space for the status line, and adapt
immediately when the window is resized. `--width` remains the maximum detail
level; use `--no-fit` when an exact surface is more important than avoiding
terminal clipping. Pipes and redirected files retain the explicitly requested
dimensions and are never resized from terminal state.

`desktop` is a high-fidelity host-screen viewer. The original desktop remains
interactive through its normal display. Glyph Forge intentionally refuses to
inject events from a terminal back into that same host display because a
synthetic key received by the focused terminal can create an input feedback
loop.

On an X11 host, Glyph Forge can launch one application in an isolated virtual
display and render that display through the same pipeline:

```bash
python -m pip install "glyph-forge[media,virtual]"
glyph-forge live launch -- xterm
glyph-forge live launch --display-width 1600 --display-height 900 -- firefox

# Explicit interactive mode (install media, virtual, and control extras)
python -m pip install "glyph-forge[media,virtual,control]"
glyph-forge live launch --control -- xterm
```

Interactive isolated mode parses UTF-8 keys and SGR mouse events without
blocking the renderer, maps only the visible glyph viewport into target pixels,
and routes events to the isolated display. Ctrl+] is an unconditional hard
stop that releases held buttons before restoring the terminal. The feature is
off unless `--control` is supplied; capture permission never implies input
permission. The child application and virtual display are closed when the
viewer exits.

## Browser Studio and sharing

```bash
glyph-forge studio

# Enable explicit one-click snapshot links on this device
glyph-forge studio --share-links

# Open Studio to trusted devices on the same LAN and enable snapshot links
glyph-forge studio --lan
```

Studio is included in the core install and opens a private loopback server. It
supports drag-and-drop images and videos, webcam capture, browser-mediated
screen capture, live style controls, and a WebGL2 glyph-atlas renderer with a
Canvas2D fallback.

Exports include PNG, scalable real-text SVG, and TXT. When supported, the Web
Share API can hand an export to another installed app. Copyable style links
encode settings only. Temporary output links are hidden unless explicitly
enabled; pressing **Copy temporary link** publishes only the current PNG frame
to the bounded in-memory local server, never the source media.

The server refuses non-loopback addresses unless trusted-LAN access is
explicitly enabled:

```bash
glyph-forge studio --lan --share-ttl 1800
```

Share an already-rendered image, audio file, or multi-gigabyte video without
copying it into memory or another directory:

```bash
# Private test link on this computer
glyph-forge share Downloads/crab-rave.glyph.mp4

# Seekable link for friends on the trusted local network
glyph-forge share Downloads/crab-rave.glyph.mp4 --lan --ttl 3600

# Override automatic LAN-address selection on a VPN or multi-NIC workstation
glyph-forge share render.mp4 --lan --advertise-host 192.168.1.42
```

File links expose exactly the selected file and support HTTP byte ranges, so a
browser can seek through large videos while Glyph Forge streams from disk. The
file is never copied or uploaded, and the link disappears when its TTL expires
or the command stops. Capability URLs are random but use unencrypted HTTP: use
`--lan` only on a trusted local network and share the URL only with intended
viewers. Glyph Forge does not provide a public Internet relay or hosted storage.
See [the sharing guide](docs/sharing.md) for behavior and security boundaries.

## Third-party extensions

Glyph Forge has a versioned plugin API for external sources, renderers,
transforms, and exporters. Installed plugin metadata is discovered without
importing plugin code; a plugin loads only when selected, inspected, or
explicitly probed. Failures are isolated per plugin.

```bash
glyph-forge plugins                 # metadata only
glyph-forge plugins --probe         # validate every installed plugin
glyph-forge plugins inspect effects # inspect one manifest

# Plugin sources and renderers use the maintained live pipeline
glyph-forge live source "plugin:synth/waves:440" --mode plugin:effects/neon
```

Plugins are ordinary Python packages registered in the
`glyph_forge.plugins` entry-point group. pipx users can install one into the
same isolated environment with `pipx inject glyph-forge PACKAGE`. Plugins run
in-process with your account's permissions and are not sandboxed, so install
only packages you trust; `GLYPH_FORGE_DISABLE_PLUGINS=1` disables automatic
discovery.

The [extension API v1 guide](docs/extensions.md) documents packaging, all four
contracts, failure isolation, viewport rules, and testing. A complete runnable
implementation is in [`examples/plugin_example.py`](examples/plugin_example.py).

## Full-screen terminal UI

```bash
python -m pip install "glyph-forge[tui]"
glyph-forge interactive
```

The TUI combines a filtered media browser, image and text previews, live
sources, output saving, runtime diagnostics, and one-key handoff to Studio. Its
preview work runs outside the UI event loop.

## Adaptive performance

Most workflows accept `--performance auto|eco|balanced|workstation`. Automatic
selection uses logical CPU count and available physical memory when those are
discoverable. The profile chooses conservative defaults for worker count,
output width, target FPS, and resampling; explicit command options always win.

Measure the renderer on the current machine with deterministic synthetic input:

```bash
glyph-forge benchmark
glyph-forge benchmark --mode braille --iterations 10
glyph-forge benchmark --performance workstation --json
```

The main data path is intentionally small:

```text
camera / video / screen / URL
              │
      newest-frame slot
              │
  normalized NumPy RGB frame
              │
   glyph / edge / subpixel renderer
       ┌──────┼─────────┐
   terminal  Studio   SVG/text
```

This architecture prevents capture speed from determining memory use. Higher
resolution still increases sampling, terminal output, and encoding costs, so
the adaptive profiles favor stable frame pacing over nominal resolution.

## Python API

The top-level package is lazy: importing `glyph_forge` does not load video or
UI backends and performs no filesystem writes.

```python
import numpy as np
from PIL import Image

from glyph_forge import FrameRenderer, RenderConfig, image_to_glyph, text_to_banner

print(text_to_banner("HELLO", font="small", style="boxed"))
print(image_to_glyph("photo.jpg", width=72, auto_scale=False))

frame = np.asarray(Image.open("photo.jpg").convert("RGB"), dtype=np.uint8)
result = FrameRenderer(
    RenderConfig(width=80, mode="braille", color="none")
).render(frame)
print(result.text)
```

For low-level capture, video export, and live presentation, the public package
also lazily exports `create_frame_source`, `LatestFramePump`,
`VideoExportConfig`, `export_glyph_video`, `InputRouter`, and
`run_terminal_session`.
[`examples/api_examples.py`](examples/api_examples.py) is a complete runnable
example that uses only generated in-memory media.

## Portability and optional behavior

- Core image/text workflows have no OpenCV, FFmpeg, display-server, or browser
  dependency.
- Optional modules are imported only when their feature is used.
- Camera indices, monitor layout, browser permissions, and available capture
  backends are controlled by the operating system.
- Screen capture under Wayland may require a desktop portal or compositor
  permission; `mss` support is backend-dependent.
- `live launch` is an X11 feature. It reports an actionable error elsewhere.
- Input injection is opt-in and depends on OS Accessibility/input-control
  permission. Safe terminal control currently targets a distinct isolated X11
  display; direct host-screen views remain viewer-only.
- ANSI and Unicode fidelity depend on terminal and font support.

## Compatibility

`imagize`, `bannerize`, and `glyphfy` remain installed for existing scripts.
They are thin adapters to the maintained `glyph-forge image` and
`glyph-forge text` implementations; there is no duplicate rendering engine.

## Development

```bash
git clone https://github.com/Ace1928/glyph_forge.git
cd glyph_forge
python -m venv .venv

# Linux/macOS
. .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --editable ".[dev,tui]"
ruff format src tests examples
ruff check src tests examples
python -m mypy src/glyph_forge
python -m pytest
python -m pytest --cov=glyph_forge --cov-fail-under=70
python -m build
python -m twine check dist/*
```

The current suite has 361 passing tests (plus one platform-dependent skip) and
measures at least 70% branch-aware coverage.
CI runs formatting, linting, type checking, Python 3.10–3.14 tests, Windows and
macOS smoke matrices, optional-extra installation, and installed-wheel resource
checks.

See [CONTRIBUTING.md](CONTRIBUTING.md), [ROADMAP.md](ROADMAP.md),
[SECURITY.md](SECURITY.md), and [CHANGELOG.md](CHANGELOG.md) for project policy
and current boundaries.

## License

Glyph Forge is available under the [MIT License](LICENSE).

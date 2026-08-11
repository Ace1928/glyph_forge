<div align="center">

# ⚡ Glyph Forge

**Images. Text. Video. Webcam. Screen. URLs — everything becomes character art.**

[![CI](https://github.com/Ace1928/glyph_forge/actions/workflows/ci.yml/badge.svg)](https://github.com/Ace1928/glyph_forge/actions/workflows/ci.yml)
[![Python 3.10–3.14](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue)](https://www.python.org/)
[![MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Play in browser](https://img.shields.io/badge/▶%20Play%20in%20browser-Studio-purple)](https://ace1928.github.io/glyph_forge/)

---

```
                                                                 
        _/_/_/  _/    _/      _/  _/_/_/    _/    _/             
     _/        _/      _/  _/    _/    _/  _/    _/              
    _/  _/_/  _/        _/      _/_/_/    _/_/_/_/               
   _/    _/  _/        _/      _/        _/    _/                
    _/_/_/  _/_/_/_/  _/      _/        _/    _/                 
                                                                 
                                                                 
```

*This banner, and every artwork below, was rendered by Glyph Forge itself.*

</div>

The same rendering engine runs everywhere: a direct CLI, a full-screen terminal
UI, a private browser Studio, and a Python API. One command renders anything,
and `glyph-forge demo` throws the whole toolkit at your terminal — memes and
video thumbnails included.

---

## ▶ Try it right now — no install

The browser Studio is the same renderer as the CLI, and it runs entirely in
your browser:

> ### **[ace1928.github.io/glyph_forge](https://ace1928.github.io/glyph_forge/)**
>
> Drag in an image or video, pick a mode, and export PNG, SVG, or text.
> WebGL2 with a full Canvas2D fallback — no GPU, no install, no account.

---

## Install on any OS

Glyph Forge supports Python 3.10–3.14 on Windows, macOS, Linux, and other
Python environments supported by its core dependencies (it runs great on
Termux/Android too).

| OS | Fastest path |
|---|---|
| **Windows** | `py -m pip install "glyphforge[all] @ https://github.com/Ace1928/glyph_forge/archive/refs/tags/v0.3.1.zip"` |
| **macOS / Linux** | `python3 -m pip install "glyphforge[all] @ https://github.com/Ace1928/glyph_forge/archive/refs/tags/v0.3.1.zip"` |
| **Any OS, isolated** | `pipx install "glyphforge[all] @ https://github.com/Ace1928/glyph_forge/archive/refs/tags/v0.3.1.zip"` |
| **Any OS, fastest** | `uv tool install "glyphforge[all] @ https://github.com/Ace1928/glyph_forge/archive/refs/tags/v0.3.1.zip"` |
| **No install at all** | `uvx --from "glyphforge[all] @ https://github.com/Ace1928/glyph_forge/archive/refs/tags/v0.3.1.zip" glyph-forge launch` |
| **Small core** | Install the same release URL without `[all]` — images, text, Studio, demo |
| **Portable binaries** | Tagged releases ship no-Python one-directory archives for all three OSes |

The Python distribution is `glyphforge` (no separator); the product and command
remain **Glyph Forge** and `glyph-forge`, and the import remains `glyph_forge`.
The hyphenated PyPI project `glyph-forge` belongs to an unrelated API client.
The versioned GitHub tag above is the canonical Python install until the
available `glyphforge` PyPI name receives its one-time owner configuration.

Install only what you use with extras:

| Extra | Adds |
|---|---|
| `tui` | Full-screen Textual interface |
| `media` | OpenCV video/webcam capture and fast MSS screen capture |
| `network` | yt-dlp URL resolution for `live url` |
| `virtual` | PyVirtualDisplay support for isolated X11 applications |
| `control` | Explicit keyboard and pointer forwarding through pynput |
| `all` | Every optional interface and backend |
| `bundle` | Maintainer tooling for standalone portable app builds |
| `dev` | Tests, Ruff, mypy, build, and release checks |

Full-colour video export also needs `ffmpeg` and `ffprobe` on `PATH`. Run
`glyph-forge doctor` for exact capability and installation guidance for your
machine.

---

## Showcase in one command

The self-contained demo needs no input files and no internet agreement — it
downloads its own assets (with offline stand-ins) and tours the entire toolkit:

```bash
glyph-forge demo

# The whole show, saved as art
glyph-forge demo --output-dir ./gallery

# Nothing leaves your machine
glyph-forge demo --offline --no-media
```

```text
━━━═◇  MEME WALL · public templates, downloaded live  ◇═━━━
▶ drake — 'me installing glyph-forge'
  Glyph Forge · braille
⣿⣿⣿⣿⣿⣿ ...
```

Every banner font, every character set, every text style, every render mode,
live memes, and the internet's most shown video frames — all rendered locally,
in less time than it takes to read this page.

**Sanity check after install:**

```bash
glyph-forge doctor   # every capability, every extra, one report
glyph-forge benchmark  # how fast is this exact machine?
```

---

## Start anywhere

```bash
# Let Glyph Forge choose the friendliest installed interface
glyph-forge launch

# Or choose explicitly
glyph-forge launch gui
glyph-forge launch tui
glyph-forge launch cli
```

`glyph-forge` with no arguments prints a compact quick start, and every
command has focused help (`glyph-forge image --help`).

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

# Choose glyph detail and final pixels independently
glyph-forge image photo.jpg --width 160 --height 90 \
  --output photo-1080p.png --output-width 1920 --output-height 1080

# Real text glyphs in an exact-size, losslessly scalable vector canvas
glyph-forge image photo.jpg --width 240 --output photo-4k.svg \
  --output-width 3840 --output-height 2160

# Browse before choosing
glyph-forge image --list-charsets
glyph-forge image --preview-charset detailed --sample photo.jpg

# FIGlet banners, fonts, styles, preview, and saving
glyph-forge text "GLYPH FORGE" --font slant --style boxed --preview
glyph-forge text --list-fonts
glyph-forge text --list-styles
glyph-forge text "SHARE THIS" --output banner.txt
```

The `--width`/`--height` pair controls the glyph grid; `--output-width` and
`--output-height` control PNG/SVG pixels without changing that grid. Supplying
one output dimension preserves the glyph canvas aspect ratio, while supplying
both produces that exact size. The output suffix selects PNG or SVG; SVG keeps
real text glyphs sharp at arbitrary zoom levels. Plain text, ANSI colour, and
HTML remain available for terminal and web output. All visual paths start from
the brighter, clearer `1.12` brightness and `1.08` contrast curve, with both
values directly overridable.

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

Install the `media` extra from the versioned source above for the primary live
backends.

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

# Video pixels are independent from sampling density too
glyph-forge video clip.mp4 output.mp4 --width 2560 --height 1440 \
  --columns 220 --rows 124 --brightness 1.18 --contrast 1.1

# Override bounded ordered workers or capture complete render metrics
glyph-forge video clip.mp4 output.mp4 --workers 8
glyph-forge video clip.mp4 output.mp4 --json > render-metrics.json
```

The exporter streams OpenCV frames directly through a vectorized glyph atlas
to FFmpeg. It does not create a temporary image sequence, and the destination
is replaced only after encoding succeeds. Hardware-adaptive workers render a
bounded number of frames concurrently, then write them in exact source order;
this scales on workstations without allowing memory use or audio timing to
drift. `--workers 1` provides the minimum-memory serial path.
Output width and height accept any even H.264 dimensions; glyph columns and
rows no longer need to divide them evenly. Divisible grids retain the direct
zero-resize hot path, while fractional cell layouts receive one final
high-quality fit to the exact requested frame.

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
python -m pip install "glyphforge[media,virtual] @ https://github.com/Ace1928/glyph_forge/archive/refs/tags/v0.3.1.zip"
glyph-forge live launch -- xterm
glyph-forge live launch --display-width 1600 --display-height 900 -- firefox

# Explicit interactive mode (install media, virtual, and control extras)
python -m pip install "glyphforge[media,virtual,control] @ https://github.com/Ace1928/glyph_forge/archive/refs/tags/v0.3.1.zip"
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
supports drag-and-drop images and videos, local text sources, webcam capture,
browser-mediated screen capture, live style controls, and presentation
fullscreen. Its WebGL2 renderer exposes the same density, edge, Braille 2×4,
quadrant 2×2, and true-colour half-block modes as the maintained live engine;
a complete Canvas2D fallback keeps those modes available without a GPU. The
same static pages power the GitHub Pages playground.

The public Studio is an installable progressive web app. On Chrome or Edge,
use **Install app**; on Safari, use **Add to Dock** or **Add to Home Screen**.
Its versioned offline shell caches only the Studio code, manifest, and brand
assets—never selected media, generated output, API responses, or temporary
shares. After one online visit, the installed editor can reopen offline and
continue processing local files and text.

Phone and tablet layouts keep a compact live preview docked above the controls,
honour notches and safe areas, use coarse-pointer touch targets of at least 44
CSS pixels, and avoid hover-only actions. Unsupported device APIs are disabled
cleanly: for example, iPhone browsers without screen capture still retain file,
text, camera (where permitted), styling, and export workflows. Adaptive defaults
start at 96, 160, or 240 glyph columns for modest, balanced, and workstation
hardware; every quality control remains manually overridable.

Output size is a separate control from glyph columns. Choose adaptive, source,
720p, 1080p, 1440p, 4K, or 8K; type an exact width and height; and toggle aspect
locking without altering glyph density. Studio fits unsafe requests to the
browser/GPU limit it actually reports instead of risking a failed canvas.
The same dimensions drive preview, PNG, sharing, and recording. SVG uses the
exact requested viewBox with real text glyphs, so it remains lossless when
scaled or zoomed.

For uploaded videos, **Render full video** restarts at frame zero and saves the
whole processed result when playback ends. Webcam and screen sources use a
manual **Record live video** toggle. When **Include source audio** is enabled,
the rendered canvas and available file, microphone, or shared-screen audio
tracks enter one browser media stream so their timestamps stay synchronized.
The browser chooses a supported WebM or MP4 encoder. Audio permission is never
requested unless the toggle is enabled, and a clear status message reports
when a source or browser provides video only.

Exports include PNG, scalable real-text SVG, and TXT. When supported, the Web
Share API can hand an export to another installed app. Copyable style links
encode settings only. Temporary output links are hidden unless explicitly
enabled; pressing **Copy temporary link** publishes only the current PNG frame
to the bounded in-memory local server, never the source media.

Fullscreen Studio output is a high-fidelity local presentation surface. Browser
screen-capture security does not permit it to inject input into the captured
desktop; use `glyph-forge live launch --control -- COMMAND` for the isolated,
interactive desktop path with its Ctrl+] emergency stop.

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

The browser surface is release-gated in Chromium, Firefox, and WebKit, plus
Pixel-class and iPhone-class touch profiles. Those checks exercise every render
mode, offline recovery, responsive geometry, accessible semantics, touch target
sizes, startup rendering latency, and a dependency-free app-shell size budget.
GitHub Pages deploys only after the complete repository CI run succeeds.

## Glyph codes — the artwork IS the link

Any image, banner, or animated GIF can be folded into one printable ASCII
string that needs no hosting, no upload, and no server. Paste it anywhere;
anyone can regenerate the original artwork from the string alone.

```bash
# Snap a shot of anything you rendered or captured
glyph-forge link code photo.jpg
# → glyph:v1:img:iVBORw0KGgoAAAANSUhEUgAAAAEA…

# Banners carry their style settings with them
glyph-forge link banner "FORGE ON" --font slant --style boxed
# → glyph:v1:banner:eyJ0ZXh0IjoiRk9SR0UgT04i…

# Peek or save it back (PNG for images, GIF for animations)
glyph-forge link decode "glyph:v1:img:…"
glyph-forge link decode "glyph:v1:…" --output restored.png
```

Round trips are lossless: an image code decodes to byte-identical data, a
banner code reproduces the exact font/style render, and a GIF code keeps every
frame plus its timing. Studio on this site and the CLI page both accept pasted
codes in their source panel. Codes embed no secrets and expire never; keep
them private if the artwork is private.

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
same isolated environment with `pipx inject glyphforge PACKAGE`. Plugins run
in-process with your account's permissions and are not sandboxed, so install
only packages you trust; `GLYPH_FORGE_DISABLE_PLUGINS=1` disables automatic
discovery.

The [extension API v1 guide](docs/extensions.md) documents packaging, all four
contracts, failure isolation, viewport rules, and testing. A complete runnable
implementation is in [`examples/plugin_example.py`](examples/plugin_example.py).

## Full-screen terminal UI

```bash
python -m pip install "glyphforge[tui] @ https://github.com/Ace1928/glyph_forge/archive/refs/tags/v0.3.1.zip"
glyph-forge interactive
```

The TUI combines a filtered media browser, image and text previews, live
sources, output saving, runtime diagnostics, and one-key handoff to Studio. Its
preview work runs outside the UI event loop.

## Adaptive performance

Most workflows accept `--performance auto|eco|balanced|workstation`. Automatic
selection uses logical CPU count and available physical memory when those are
discoverable. The profile chooses conservative defaults for worker count,
output width, target FPS, and resampling. Video export uses those workers as a
bounded ordered pipeline; explicit command options always win.

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
 downsample → cached tone LUT
              │
   glyph / edge / subpixel renderer
       ┌──────┼─────────┐
   terminal  Studio   SVG/text
```

This architecture prevents capture speed from determining memory use. Tone
mapping happens after downsampling through a cached 256-entry lookup table, so
clearer defaults do not add full-frame per-pixel math to native live and video
paths. Higher glyph density still increases sampling, terminal output, and
encoding costs, so adaptive profiles favor stable frame pacing over nominal
resolution. Final pixel size remains independently selectable.

## Python API

The top-level package is lazy: importing `glyph_forge` does not load video or
UI backends and performs no filesystem writes.

```python
from pathlib import Path

import numpy as np
from PIL import Image

from glyph_forge import (
    FrameRenderer,
    RenderConfig,
    image_to_glyph,
    render_text_png,
    render_text_svg,
    text_to_banner,
)

print(text_to_banner("HELLO", font="small", style="boxed"))
print(image_to_glyph("photo.jpg", width=72, auto_scale=False))

frame = np.asarray(Image.open("photo.jpg").convert("RGB"), dtype=np.uint8)
result = FrameRenderer(
    RenderConfig(width=80, mode="braille", color="none")
).render(frame)
print(result.text)

render_text_png(result.text, output_width=1920, output_height=1080).save("frame.png")
Path("frame.svg").write_text(
    render_text_svg(result.text, output_width=3840, output_height=2160),
    encoding="utf-8",
)
```

For low-level capture, video export, and live presentation, the public package
also lazily exports `create_frame_source`, `LatestFramePump`,
`VideoExportConfig`, `VideoExportResult`, `export_glyph_video`, `InputRouter`, and
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
ruff format src tests examples tools
ruff check src tests examples tools
python -m mypy src/glyph_forge
python -m pytest
python -m pytest --cov=glyph_forge --cov-fail-under=70
python -m build
python -m twine check dist/*
```

The current suite has 429 passing tests (plus one platform-dependent skip) and
measures at least 70% branch-aware coverage.
CI runs formatting, linting, type checking, Python 3.10–3.14 tests, Windows and
macOS smoke matrices, optional-extra installation, and installed-wheel resource
checks.

See [CONTRIBUTING.md](CONTRIBUTING.md), [ROADMAP.md](ROADMAP.md),
[SECURITY.md](SECURITY.md), and [CHANGELOG.md](CHANGELOG.md) for project policy
and current boundaries.

## License

Glyph Forge is available under the [MIT License](LICENSE).

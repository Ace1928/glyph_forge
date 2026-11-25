# Glyph Forge

[![CI](https://github.com/Ace1928/glyph_forge/actions/workflows/ci.yml/badge.svg)](https://github.com/Ace1928/glyph_forge/actions/workflows/ci.yml)
[![Python 3.10-3.13](https://img.shields.io/badge/Python-3.10--3.13-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Adaptive image, text, video, and live-stream rendering for terminals.

Glyph Forge turns pixels into plain text, ANSI 16/256-color output, truecolor
terminal art, HTML, SVG, and shareable media. Its defaults adapt to the actual
output stream: a modern interactive terminal gets rich output, while redirected
output, `NO_COLOR`, `TERM=dumb`, and non-Unicode environments fall back safely.

The package is self-contained. It does not depend on Eidosian Forge or any
sibling repository, does not send telemetry, and does not download sample media
during normal use.

## Quick start

Python 3.10 or newer is required. Install from a checkout:

```bash
git clone https://github.com/Ace1928/glyph_forge.git
cd glyph_forge
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install .
```

Then try the asset-free demo:

```bash
glyph-forge demo
glyph-forge bannerize "GLYPH FORGE" --font slant --style boxed
glyph-forge imagize convert photo.jpg --width 100 --color ansi256
```

`--color auto` is accepted by streaming. For single-image conversion, choose
`none`, `ansi16`, `ansi256`, or `truecolor`; `none` is the portable default.

## What it can do

| Input | Output | Installation |
| --- | --- | --- |
| Text | FIGlet banners with styles and optional ANSI color | Core |
| Images and PIL objects | Plain glyphs, ANSI 16/256, truecolor, HTML | Core |
| Local image directories | HTML, SVG, and PNG galleries | Core |
| Video files and webcams | Adaptive terminal glyph streams | `.[streaming]` |
| Screen capture and URLs | Buffered terminal streams and recordings | `.[streaming]` plus system tools |
| Interactive workflow | Textual terminal interface | `.[tui]` |

Core conversion uses NumPy and Pillow. OpenCV, yt-dlp, Textual, Playwright,
audio backends, and clipboard support are isolated in optional extras.

```bash
python -m pip install ".[streaming]"
python -m pip install ".[tui]"
python -m pip install ".[full]"
glyph-forge doctor
```

`ffmpeg` and `ffplay` are external programs, not Python packages. Glyph Forge
reports whether they are available but never installs them silently.

## Image conversion

```bash
# Fast, portable text output
glyph-forge imagize convert photo.jpg --width 96 --color none

# Detailed terminal color
glyph-forge imagize convert photo.jpg \
  --width 120 \
  --color truecolor \
  --dither \
  --dither-algorithm atkinson \
  --edge-enhance

# Write instead of printing
glyph-forge imagize convert photo.jpg --output portrait.txt --color none
```

The command validates paths and options before conversion and returns a nonzero
exit code on failure. ANSI sequences are never added to the default plain-text
mode.

## Streaming

```bash
glyph-forge stream video.mp4
glyph-forge stream video.mp4 --mode ascii --color none --no-audio
glyph-forge stream --webcam 0 --resolution 480p --fps 24
glyph-forge stream "https://www.youtube.com/watch?v=VIDEO_ID" --yes
```

The default `--mode auto --color auto` inspects terminal encoding, color depth,
TTY state, `NO_COLOR`, and `TERM`. Network sources and potentially sensitive
screen capture require confirmation unless the caller supplies an explicit
answer. Redirected streams do not emit cursor-control sequences.

## Local galleries

Glyph Forge does not rely on a repository-owned media library. Point it at your
own directory:

```bash
glyph-forge gallery ./photos --output-dir ./gallery --formats html,svg,png
```

An existing non-empty output directory is refused unless `--overwrite` is
provided. Source names are escaped before being embedded in the generated HTML.

## Python API

```python
from PIL import Image

from glyph_forge import image_to_glyph

image = Image.open("photo.jpg")
art = image_to_glyph(
    image,
    width=80,
    auto_scale=False,
    color_mode="ansi256",
    dithering=True,
)
print(art)
```

For repeated operations, reuse the high-level API:

```python
from glyph_forge import get_api

forge = get_api()
print(forge.generate_banner("HELLO", font="slant", style="boxed"))
print(forge.image_to_glyph("photo.jpg", width=80, color_mode="none"))
```

Public failures use the `GlyphForgeError` hierarchy; failed conversions do not
return strings that merely look like successful art. The legacy
`image_to_Glyph()` spelling remains as a deprecated 1.x compatibility alias.

## Terminal compatibility

- Linux, macOS, and Windows are tested in CI.
- Python 3.10, 3.11, 3.12, and 3.13 are covered.
- UTF-8, ASCII fallback, TTY and redirected output are first-class modes.
- `NO_COLOR` disables color; `FORCE_COLOR=1` permits color in non-TTY logs.
- `GLYPH_FORGE_ASCII=1` forces ASCII glyphs.
- `GLYPH_FORGE_CACHE_DIR` changes the streaming lookup-table cache.

See [Terminal compatibility](docs/TERMINALS.md) for the complete selection and
downgrade rules.

## Documentation

- [API reference](docs/API.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Configuration](docs/CONFIGURATION.md)
- [Terminal compatibility](docs/TERMINALS.md)
- [Performance and benchmarking](docs/PERFORMANCE.md)
- [Development guide](docs/DEVELOPMENT.md)
- [Security policy](SECURITY.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## License

Glyph Forge is available under the [MIT License](LICENSE).

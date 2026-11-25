# Python API reference

This document covers the supported high-level API. Internal modules may evolve
between minor releases unless they are re-exported from `glyph_forge`.

## One-shot image conversion

```python
from glyph_forge import image_to_glyph

art = image_to_glyph(
    "photo.png",
    width=100,
    color_mode="none",
    dithering=True,
    dither_algorithm="atkinson",
)
```

`image_to_glyph(image_path, output_path=None, style=None, color_mode="none",
**options) -> str` accepts a string path, `os.PathLike`, or `PIL.Image.Image`.
The recognized constructor options are:

- `charset`: built-in alphabet name or a custom character string.
- `width`, `height`: output dimensions in characters.
- `invert`: reverse the density map.
- `brightness`, `contrast`, `gamma`: tone controls.
- `auto_scale`: fit output to terminal dimensions when true.
- `dithering`, `dither_algorithm`: `none`, `floyd-steinberg`, or `atkinson`.
- `aspect_ratio`: terminal character aspect correction.
- `resample`: `nearest`, `bilinear`, `bicubic`, or `lanczos`.
- `autocontrast`, `equalize`, `invert_image`, `edge_enhance`, `sharpen`:
  optional Pillow filters.
- `blur_radius`, `posterize_bits`: additional style controls.
- `threads`: row conversion workers; zero selects a CPU-aware default.

Color modes are `none`, `ansi`, `ansi16`, `ansi256`, `truecolor`, and `html`.
`ansi` is a truecolor alias. `rgb` and `web` remain compatibility aliases for
`truecolor` and `html`.

When `output_path` names a file, the result and a metadata sidecar are written.
When it names a directory, Glyph Forge generates a collision-resistant filename.

## Reusable high-level API

`get_api() -> GlyphForgeAPI` returns a process-wide, lazily initialized API
object. Construction loads configuration but does not initialize the image
converter until the first image request.

### `GlyphForgeAPI.generate_banner`

```python
banner = forge.generate_banner(
    text="GLYPH FORGE",
    style="boxed",
    font="slant",
    width=100,
    effects=None,
    color=False,
)
```

Returns a string. Unknown fonts and invalid style settings raise `ValueError`.

### `GlyphForgeAPI.image_to_glyph`

```python
art = forge.image_to_glyph(
    "photo.jpg",
    width=80,
    height=None,
    charset="general",
    color_mode="ansi256",
)
```

This method exposes the same conversion controls as `ImageGlyphConverter` and
reuses the configured converter when no per-call overrides require a temporary
one.

`image_to_Glyph(*args, **kwargs)` is a deprecated compatibility alias. It emits
`DeprecationWarning` and will not be removed during the 1.x release line.

### Discovery methods

- `get_available_fonts() -> list[str]`
- `get_available_styles() -> dict[str, dict[str, object]]`
- `get_available_alphabets() -> list[str]`
- `preview_font(font, text="Glyph Forge") -> str`
- `preview_style(style, text="Glyph Forge") -> str`

### `save_to_file`

`save_to_file(glyph_art, file_path) -> bool` creates parent directories and
returns whether the write succeeded. New code that needs the underlying error
should write explicitly with `pathlib.Path.write_text` or use a conversion
method with `output_path`, whose failure raises an exception.

## Streaming renderer

Install the streaming extra before using this surface:

```bash
python -m pip install ".[streaming]"
```

```python
import cv2

from glyph_forge import GlyphRenderer, RenderConfig

frame = cv2.imread("frame.png")  # OpenCV BGR array
renderer = GlyphRenderer(
    RenderConfig(mode="ascii", color="none", dithering=False)
)
print(renderer.render(frame, width=100, height=40))
```

`RenderConfig` validates mode, color, gamma, and contrast at construction.
Supported modes are `ascii`, `blocks`, `braille`, `gradient`, and `hybrid`.
Streaming color supports `none`, `ansi256`, and `truecolor`.

`GlyphRenderer.render` accepts grayscale, BGR, or BGRA NumPy arrays. Empty or
malformed frames, non-positive dimensions, and unsupported modes raise clear
`TypeError` or `ValueError` exceptions before OpenCV work begins.

## Terminal detection

```python
from glyph_forge import detect_terminal

capabilities = detect_terminal()
print(capabilities.preferred_color)
print(capabilities.resolve_render_mode("auto"))
```

`TerminalCapabilities` is immutable and contains TTY state, dimensions, color
depth, Unicode and hyperlink support, encoding, and `TERM`. `as_dict()` returns
a JSON-serializable snapshot.

## Exceptions

All package-defined exceptions inherit from `GlyphForgeError`:

- `GlyphForgeInputError`: invalid input or option.
- `GlyphForgeRenderError`: decode, render, or output failure.
- `GlyphForgeConfigError`: invalid or unwritable configuration.
- `GlyphForgeDependencyError`: missing optional dependency.

Python built-in exceptions remain in use for low-level programming errors where
wrapping would hide useful information.

## Versioning

`glyph_forge.__version__` is the string release version. `glyph_forge.VERSION`
is the corresponding integer tuple.

# Rendering contract v1

Glyph Forge 0.4 routes still images from the CLI, TUI, Python API, and legacy
compatibility entry points through one immutable request and one renderer. This
page is the behavioral contract for applications, presets, and future project
files.

## Minimal API

```python
from glyph_forge import RenderRequest, render_image

request = RenderRequest(
    width=160,
    mode="braille",
    output_format="png",
    output_width=1920,
    output_height=1080,
    fit="contain",
)
artifact = render_image("photo.jpg", request, destination="photo.png")

print(artifact.columns, artifact.rows)
print(artifact.pixel_width, artifact.pixel_height)
print(artifact.metrics.to_dict())
```

`RenderRequest.to_dict()` returns a JSON-compatible object containing
`contract_version: 1`. `RenderRequest.from_dict()` validates that object and
rejects unknown fields, unsupported versions, invalid types, non-finite tone
values, and unsafe dimensions with `RenderContractError`.

## Geometry has two independent layers

- `width` and optional `height` are character cells. They determine artistic
  detail and renderer cost.
- `output_width` and `output_height` are final PNG/SVG pixels. They never add
  glyph detail and are rejected for text, ANSI, truecolor, or HTML output.
- Supplying one pixel axis derives the other from the intrinsic glyph canvas.
- Supplying both produces that exact canvas.
- `fit="contain"` preserves all art, `cover` fills and clips, and `stretch`
  maps both edges exactly. `alignment` selects one of nine anchors.
- SVG contains real text and a clipping transform, so it stays sharp when
  zoomed. PNG is the raster equivalent at the requested pixel dimensions.

The CLI shorthand is:

```bash
glyph-forge image photo.jpg --width 160 --height 90 \
  --output photo.png --size 1920x1080 --fit-mode contain --align center
```

## Formats and modes

`output_format` accepts `text`, `ansi256`, `truecolor`, `html`, `png`, and
`svg`. The maintained modes are `glyph`, `edge`, `braille`, `half-block`, and
`quadrant`. PNG/SVG accept every maintained mode. HTML currently uses glyph
mode because its safe encoder represents per-glyph truecolor spans.

File suffixes select `.png`, `.svg`, `.html`/`.htm`, `.ansi`, or plain text in
the CLI and convenience API. Saves use a same-directory temporary file,
flush data, atomically replace the destination, and clean up after failure.

## Tone, alpha, and character sets

The default curve is brightness `1.12`, contrast `1.08`. Both range from 0.0
to 2.0. Tone adjustment uses the same cached 256-value lookup table for native
stills, live frames, and video. Browser conformance fixtures use the same
rounding and integer RGB luma weights.

Transparent sources are composited against `background` before sampling.
EXIF orientation is applied during load. Sources are bounded to 100 megapixels;
cell dimensions are bounded to 4096 and output axes to 32768 in the public
contract. User interfaces may choose smaller device-safe limits.

Use a named density, special, or language preset as `charset`. A likely typo
such as `detaled` fails with suggestions. Prefix an intentional lowercase-only
custom sequence with `literal:`, for example `literal:abcdef`. Character
sequences containing spaces, symbols, or uppercase characters remain accepted
directly for compatibility.

## Results and errors

`RenderArtifact` contains:

- `glyph_text`, plus encoded `data` as `str` or `bytes`;
- media type and conventional suffix;
- cell and optional pixel dimensions;
- the normalized request;
- source geometry, output bytes, and load/render/encode/total milliseconds.

Callers can handle the typed hierarchy:

```python
from glyph_forge import (
    GlyphForgeRenderError,
    RenderContractError,
    RenderExecutionError,
    RenderExportError,
    SourceLoadError,
)
```

Contract failures occur before decoding. Source, render, encode, and save
failures stay distinguishable. The CLI converts them to concise option errors;
the Python API leaves them typed for programmatic recovery.

## Performance and conformance

The native path decodes once, downsamples before tone mapping, performs pixel
math in NumPy, and emits an artifact without crossing the legacy stateful
converter. Measure the complete path separately from the kernel:

```bash
glyph-forge benchmark --pipeline --performance eco --json
glyph-forge benchmark --kernel --mode braille --iterations 10
```

`tests/fixtures/render-contract-v1.json` is consumed by Python and JavaScript.
It locks all five modes, integer color luma, tone rounding, subcell bit order,
truecolor, and Sobel edge direction/weight. Changes to these semantics require
a new contract version or an explicitly documented compatible extension.

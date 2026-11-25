# Architecture

Glyph Forge is organized around a small core and opt-in media adapters. This
keeps image and banner conversion portable while allowing video, browser,
capture, audio, and TUI features to use specialized dependencies.

```text
CLI / Python callers
        |
        +-- High-level API -------- banner and image services
        |                                  |
        |                                  +-- NumPy / Pillow transforms
        |                                  +-- text / ANSI / HTML output
        |
        +-- Streaming engine ------ source extractors and capture adapters
                                           |
                                           +-- adaptive frame buffer
                                           +-- vectorized glyph renderer
                                           +-- terminal display
                                           +-- optional recorder / audio sync
```

## Package boundaries

| Package | Responsibility |
| --- | --- |
| `glyph_forge.api` | Stable, reusable high-level client |
| `glyph_forge.services` | Complete user operations such as image and video conversion |
| `glyph_forge.transformers` | Focused image, edge, depth, and color transforms |
| `glyph_forge.renderers` | Plain text, ANSI, HTML, and SVG serialization |
| `glyph_forge.streaming` | Extraction, buffering, frame rendering, recording, and synchronization |
| `glyph_forge.terminal` | Side-effect-free terminal capability selection |
| `glyph_forge.config` | Layered configuration and atomic persistence |
| `glyph_forge.cli` | Typer command adapters and human-readable diagnostics |
| `glyph_forge.ui` | Optional Textual interface |

The package root uses lazy attributes. `import glyph_forge` does not import
OpenCV, create configuration or cache directories, configure logging, or install
signal handlers.

## Image pipeline

1. Validate the source type and open it through Pillow.
2. Determine character dimensions and apply terminal scaling when requested.
3. Resize with the selected resampling filter.
4. Apply deterministic luminance and optional image filters.
5. Map 8-bit luminance through a precomputed density lookup table.
6. Run-length encode repeated color sequences where the output format permits.
7. Return the string, then atomically select an output name and metadata sidecar
   when persistence was requested.

Errors at the public conversion boundary become `GlyphForgeRenderError` or
`GlyphForgeInputError`; they never become plausible-looking result strings.

## Streaming pipeline

The unified engine separates acquisition from display with an adaptive,
thread-safe buffer:

```text
source -> extractor/capture -> frame producer -> adaptive buffer
                                              -> renderer -> display
                                                          -> recorder
```

OpenCV frames use BGR byte order. The renderer resizes frames, computes
perceptual luminance through an sRGB-to-linear lookup table, applies optional
tone mapping and ordered dithering, and selects glyphs with vectorized NumPy
indexing.

ANSI 256-color matching uses a precomputed RGB-to-palette table. Tables are
stored in a compressed NumPy archive loaded with `allow_pickle=False`. Cache
writes use a temporary file, flush, `fsync`, and atomic replacement. Corrupt or
incomplete caches are ignored and rebuilt; user-writable cache data is never
deserialized as executable Python objects.

## Terminal adaptation

Capability detection happens before streaming configuration is constructed.
The selection considers TTY state, encoding, environment standards, and color
depth. Unsupported explicit requests are predictably downgraded. Non-interactive
streams receive sequential frames without clear-screen or cursor-home control
sequences.

The detection object is independent from Rich and OpenCV, which makes it usable
in applications embedding Glyph Forge.

## Configuration

Settings are four distinct layers:

```text
built-in defaults < system JSON < user JSON < runtime overrides
```

Reads do not create files. Runtime overrides are never serialized. User writes
store only user overrides using atomic replacement and restrictive POSIX file
permissions. System configuration is read-only inside Glyph Forge.

## Concurrency and lifecycle

- High-level API initialization is protected by a lock.
- Configuration reads and writes use a reentrant lock and defensive copies.
- Streaming uses explicit stop events and bounded/adaptive buffering.
- Subprocesses are invoked without a shell and are terminated during cleanup.
- Optional components load when requested, not during package import.

## Compatibility contract

The root exports and the documented high-level API are the stable surface for
the 1.x line. Internal extractor, capture, and TUI types may evolve as platform
APIs change. Deprecations remain available for at least the 1.x line and emit a
standard Python warning before removal in a future major release.

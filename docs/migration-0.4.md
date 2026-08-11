# Migrating to Glyph Forge 0.4

Version 0.4 converges still-image behavior without abruptly removing the 0.x
surface. Existing commands and imports continue to work, while new code gains
typed requests, structured artifacts, exact sizing, and predictable failures.

## Recommended replacements

| Compatibility surface | Preferred 0.4 surface | Planned removal |
|---|---|---|
| `ImageGlyphConverter(...)` | `RenderRequest(...)` + `render_image(...)` | 1.0 |
| `GlyphForgeAPI.image_to_Glyph(...)` | `GlyphForgeAPI.image_to_glyph(...)` | 1.0 |
| top-level `get_config(profile)` | `get_profile_config(profile)` | 1.0 |
| persistent config through ambiguous names | `get_settings()` | Stable |
| direct PNG/SVG post-scaling | request `output_width`/`output_height`/`fit` | Stable |

Deprecated Python calls emit `DeprecationWarning`, as conventional for library
migrations. They remain tested through the 0.x line. The installed `imagize`,
`bannerize`, and `glyphfy` commands remain thin compatibility launchers and do
not maintain separate render engines.

## Stateful converter to immutable request

Before:

```python
from glyph_forge.services.image_to_glyph import ImageGlyphConverter

converter = ImageGlyphConverter(width=120, charset="detailed")
text = converter.convert("photo.jpg")
```

After:

```python
from glyph_forge import RenderRequest, render_image

request = RenderRequest(width=120, charset="detailed")
artifact = render_image("photo.jpg", request)
text = artifact.glyph_text
```

The modern call raises a `GlyphForgeRenderError` subtype instead of returning
an error sentence. This prevents a failed conversion from being mistaken for
valid artwork.

## Exact output size

Character resolution and final pixels are deliberately separate:

```bash
glyph-forge image photo.jpg --width 160 --height 90 \
  --output result.svg --size 3840x2160 --fit-mode contain
```

Python uses the corresponding request fields. Existing `--output-width` and
`--output-height` options remain supported and can specify only one derived
axis. `--size` cannot be mixed with them. Pixel sizing requires PNG or SVG.

## Character-set typo safety

Named density, special, and language presets now share one resolver. If an
intentional custom sequence looks like a misspelled lowercase preset, prefix
it with `literal:`:

```python
RenderRequest(charset="literal:abcdef")
```

Symbolic sequences such as ` .:-=+*#%@` continue to work directly.

## Configuration

`get_settings()` returns a layered, thread-safe manager. System defaults are
read-only, user writes are schema-versioned and atomic, and runtime overrides
are session-only. Configuration paths follow APPDATA on Windows, Application
Support on macOS, and XDG on Linux; `GLYPH_FORGE_CONFIG_HOME` and
`GLYPH_FORGE_CONFIG_FILE` remain explicit overrides. Unversioned 0.x JSON is
read and rewritten as schema version 1 on the next successful user update.
When the new canonical file is absent, 0.4 also discovers the former
`%APPDATA%/GLYPH_Forge/user_config.json` path on Windows and the former
XDG-style `glyph_forge/user_config.json` path on macOS. The next successful
user update writes the complete settings to the canonical location; the old
file is deliberately retained as a recovery copy. Explicit path overrides
never trigger legacy discovery.

Failed writes roll memory back to the last persisted state, so a caller never
observes an update that was not saved.

The optional Eidos design profile now lives beside the canonical platform
configuration and uses the same atomic writer. A profile found at the legacy
`glyph-forge/eidos_profile.yml` path is still read and is migrated on its next
update; the legacy file is left untouched as a recovery copy.

## Contract stability

Serialized requests include `contract_version`. Version 0.4 accepts version 1
only. Additive application metadata should be stored beside the request rather
than inserted into it, because unknown request keys are rejected. See
[rendering.md](rendering.md) for every field, format, result, and error.

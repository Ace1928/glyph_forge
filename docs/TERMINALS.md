# Terminal compatibility

Glyph Forge treats terminal capabilities as input data, not assumptions.

## Detection inputs

`detect_terminal()` considers:

- whether the output stream is a TTY;
- stream encoding;
- terminal dimensions;
- `TERM` and `COLORTERM`;
- the community-standard `NO_COLOR` variable;
- `FORCE_COLOR` for explicit non-TTY color output;
- `GLYPH_FORGE_ASCII` for an explicit character-set fallback;
- common terminal identifiers for hyperlink support.

Detection does not write probe characters or control codes to the stream.

## Color selection

| Environment | Adaptive selection |
| --- | --- |
| Redirected output | No color |
| `NO_COLOR` present | No color |
| `TERM=dumb` | No color |
| Interactive basic terminal | ANSI 16 where supported by the operation |
| `TERM` contains `256color` | ANSI 256 |
| `COLORTERM=truecolor` or `24bit` | Truecolor |

Streaming supports no color, ANSI 256, and truecolor. An explicit truecolor
request is downgraded to ANSI 256 or no color when necessary. Single-image
conversion additionally supports ANSI 16.

`FORCE_COLOR=1` permits color when output is not a TTY. `NO_COLOR` and
`TERM=dumb` still take precedence because they communicate a safety constraint.

## Character selection

UTF-capable streams use the density-sorted Unicode gradient by default. The
renderer uses the ASCII gradient `@%#*+=-:. ` when:

- `TERM=dumb`;
- the stream encoding cannot represent block and Braille characters; or
- `GLYPH_FORGE_ASCII=1` is set.

Explicit Braille, blocks, gradient, and hybrid requests downgrade to ASCII when
Unicode is unavailable.

## Cursor control

Interactive streams may use clear-screen and cursor-home sequences to update a
frame in place. Redirected output receives complete frames separated by a
newline. This keeps logs and pipelines free of cursor movement codes.

## Narrow terminals

Terminal size is read through portable Python APIs with an 80 by 24 fallback.
The high-level image converter preserves aspect ratio and caps output to the
available rows and columns when `auto_scale=True`. Streaming reserves lines for
status output.

## Application integration

Applications may inspect and resolve settings without using the CLI:

```python
from glyph_forge import detect_terminal

caps = detect_terminal()
mode = caps.resolve_render_mode("auto")
color = caps.resolve_color("auto", streaming=True)
```

Pass a different text stream and environment mapping to `detect_terminal` for
tests or for output directed somewhere other than `sys.stdout`.

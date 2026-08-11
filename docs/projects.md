# Projects, presets, and batch queues

Glyph Forge project files make a creative session portable and recoverable.
They store a relative source reference and one or more non-destructive render
variants. Presets carry the exact same canonical `RenderRequest` without any
source media, so settings can be shared safely between the CLI, TUI, Studio,
and Python API.

## Quick start

Create a project from media anywhere on the machine:

```bash
glyph-forge project new artwork.glyphforge.json ~/Pictures/photo.jpg
glyph-forge project info artwork.glyphforge.json
glyph-forge project render artwork.glyphforge.json --output artwork.txt
```

An external source is copied to `assets/` beside the project. Use
`--reference-only` only when the source is already within the project
directory; Glyph Forge rejects absolute paths and parent traversal rather than
creating a project that breaks when shared.

Variants retain alternative looks without changing the source or another
variant:

```bash
glyph-forge project variant-add artwork.glyphforge.json poster --name Poster
glyph-forge project variant-select artwork.glyphforge.json poster
glyph-forge project variant-remove artwork.glyphforge.json poster
```

Export, inspect, apply, and render portable presets:

```bash
glyph-forge preset export artwork.glyphforge.json poster.glyphpreset.json
glyph-forge preset info poster.glyphpreset.json
glyph-forge preset apply poster.glyphpreset.json another.glyphforge.json \
  --new-variant poster
glyph-forge preset render poster.glyphpreset.json photo.jpg --output poster.svg
```

Create a preset directly when automation needs explicit values:

```bash
glyph-forge preset create "4K poster" poster.glyphpreset.json \
  --mode braille --width 240 --height 135 --format svg \
  --output-width 3840 --output-height 2160
```

Run one preset over a bounded queue. Every destination is atomic, failures are
isolated by default, and duplicate source stems receive stable numbered names:

```bash
glyph-forge batch poster.glyphpreset.json images/*.jpg \
  --output-dir exports --workers 4
glyph-forge batch poster.glyphpreset.json images/*.jpg \
  --output-dir exports --json > batch-report.json
```

## TUI and browser Studio

Run `glyph-forge interactive` and open the **Project** tab for the same
workflow without composing commands. The TUI can create or open a project,
select and edit variants, undo and redo, import or export presets, and run or
cancel a bounded image queue. Opening a project hydrates the existing image
controls and preview; preview changes update the active variant through the
same autosaving `ProjectSession` used by the API.

The installable browser Studio offers **Keep a project** after a local image or
video is selected. It provides local autosave and startup recovery, a bounded
recent list, non-destructive variants, undo and redo, project and preset JSON
downloads, direct project/preset file launch, and a collision-safe sequential
image queue. `Ctrl`/`Command`+`S` saves a project; `Ctrl`/`Command`+`Z`,
`Shift`+`Ctrl`/`Command`+`Z`, and `Ctrl`/`Command`+`Y` operate project history
when focus is outside an editable field.

Browsers never gain silent filesystem access. A Studio project therefore
references `assets/<source-name>` but cannot copy the selected source into that
folder automatically. When sharing the download, place the source at that
relative path beside the JSON. When reopening it, reselect the referenced
media after the project settings load. Source bytes are not stored in browser
autosave, recent history, presets, or the offline app shell.

Native CLI and TUI batches use bounded worker concurrency. Studio processes
one image at a time because browsers gate automatic downloads; it still caps
the queue at 1,000 files, isolates decode/export failures, supports
cancellation, and gives colliding names stable numeric suffixes.

## Recovery and history

Long-lived interfaces use `ProjectSession`, which provides:

- a default 100-operation undo/redo history with a hard configurable bound;
- debounced, atomic autosave sidecars next to the project;
- explicit dirty and autosave-error state for reliable UI status;
- recovery only when the saved project still has the exact base SHA-256;
- refusal to apply a stale sidecar over a project changed by another process;
- atomic promotion of recovered work and explicit discard.

The CLI renders a compatible recovery by default. Promote or discard it
explicitly with:

```bash
glyph-forge project recover artwork.glyphforge.json
glyph-forge project recover artwork.glyphforge.json --discard
```

`glyph-forge project recent` reads the bounded, platform-native recent list;
`--prune` removes paths that no longer exist. Recent history is user-local and
is never embedded in a portable project.

## Python API

The high-level API and standalone contracts use the same implementation:

```python
from glyph_forge import GlyphForgeAPI, RenderPreset, RenderRequest

api = GlyphForgeAPI()
project = api.create_project(
    "work/art.glyphforge.json",
    "work/assets/photo.jpg",
    request=RenderRequest(width=160, mode="braille"),
)

with api.open_project("work/art.glyphforge.json") as session:
    session.add_variant("bright", "Bright")
    session.update_active_request(
        session.project.active.request.with_updates(brightness=1.25)
    )
    session.save()

artifact = api.render_project(
    "work/art.glyphforge.json",
    variant="bright",
    destination="work/bright.txt",
)

report = api.render_batch(
    ["work/assets/photo.jpg"],
    "work/exports",
    RenderPreset("Bright", artifact.request),
    workers=1,
)
```

Lower-level applications can import `GlyphProject`, `RenderVariant`,
`AssetReference`, `ProjectSession`, `RecentProjectStore`, `RenderPreset`,
`BatchRenderItem`, `CancellationToken`, and `render_batch` from `glyph_forge`.

## Versioned document contracts

Project and preset roots are strict JSON objects. Unknown fields are rejected
so misspellings never silently alter output. Documents are limited to 4 MiB,
projects to 256 variants, and metadata to bounded finite JSON values.
Python and browser implementations share a checked-in golden project fixture,
so a document accepted and normalized by one runtime is continuously verified
against the other.

A version-one project contains:

```json
{
  "schema": "glyph-forge-project",
  "schema_version": 1,
  "name": "Artwork",
  "source": {"kind": "image", "path": "assets/photo.jpg"},
  "variants": [
    {"id": "default", "name": "Default", "request": {"contract_version": 1}}
  ],
  "active_variant": "default",
  "created_at": "2026-08-11T10:00:00.000Z",
  "updated_at": "2026-08-11T10:00:00.000Z",
  "metadata": {}
}
```

The abbreviated request above is explanatory; real documents contain every
field returned by `RenderRequest.to_dict()`. A preset root uses
`"schema": "glyph-forge-preset"`, its own `schema_version: 1`, `name`, one
complete `request`, and `metadata`.

Asset paths use NFC-normalized forward slashes. Absolute paths, drive prefixes,
`.`/`..`, Windows reserved device names, control characters, and characters
illegal on common desktop filesystems are rejected. Loading a project does not
require its asset to be online, which lets users relocate or restore media;
rendering reports a typed source error if it is still missing.

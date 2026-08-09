# Glyph Forge extension API v1

Glyph Forge plugins add media sources, frame renderers, transforms, and
exporters without modifying or duplicating the core pipeline. Discovery uses
Python package entry points and is lazy: listing metadata does not import plugin
code. Only an explicitly selected plugin, `plugins inspect`, or `plugins
--probe` loads third-party code.

Plugins execute in-process with the user's permissions; this API is an
extension boundary, not a security sandbox. Install only packages you trust.
Python-level load and execution failures are isolated and reported per plugin,
but a malicious native extension can still crash or block its host process. Set
`GLYPH_FORGE_DISABLE_PLUGINS=1` to disable automatic entry-point discovery;
in-process registrations remain available to an embedding application.

## Minimal package

Export a `PluginManifest` object or a zero-argument function returning one:

```python
# src/my_glyph_plugin/__init__.py
from glyph_forge.plugins import PluginManifest, RenderOutput


class Renderer:
    def render(self, frame, *, max_width=None, max_height=None):
        width = min(max_width or 8, 8)
        height = min(max_height or 2, 2)
        return RenderOutput("\n".join("@" * width for _ in range(height)), width, height)


def plugin():
    return PluginManifest(
        name="My glyph effects",
        version="1.0.0",
        renderers={"solid": lambda request: Renderer()},
    )
```

Register that function in the plugin package's `pyproject.toml`:

```toml
[project]
name = "my-glyph-plugin"
version = "1.0.0"
dependencies = ["glyph-forge"] # Pin to the first release your plugin tests.

[project.entry-points."glyph_forge.plugins"]
my-effects = "my_glyph_plugin:plugin"
```

The entry-point name (`my-effects`) is the stable plugin identifier. Component
names are lowercase and local to that plugin. Together they form the qualified
reference `my-effects/solid`.

After installing into the same Python environment as Glyph Forge:

```bash
glyph-forge plugins
glyph-forge plugins inspect my-effects
glyph-forge demo --mode plugin:my-effects/solid
glyph-forge image photo.jpg --mode plugin:my-effects/solid
glyph-forge live source camera:0 --mode plugin:my-effects/solid
```

For a pipx installation, inject the plugin into Glyph Forge's environment:

```bash
pipx inject glyph-forge my-glyph-plugin
```

## Source contract

A source contribution maps a component name to a callable accepting one
`SourceRequest`. It returns an object with:

- a non-empty `name` property;
- `read()`, returning the next grayscale, RGB, or RGBA NumPy-compatible frame,
  or `None` at end of stream;
- an idempotent `close()` method.

Use it through `plugin:PLUGIN/SOURCE:RESOURCE`. The resource suffix is opaque
and preserves additional colons, allowing URLs and structured identifiers.
Capture width, height, frame rate, and loop intent are passed separately in the
request. Returned frames enter the standard normalization and newest-frame pump,
so buffering remains bounded.

## Renderer contract

A renderer contribution maps a component name to a factory accepting one
`RendererRequest`. Construct expensive state in the factory, not for every
frame. Its returned object implements:

```python
render(frame, *, max_width=None, max_height=None) -> RenderOutput
```

Frames are contiguous uint8 RGB arrays. A renderer must honor non-null viewport
limits and return text, positive logical dimensions, and exactly `height` rows.
Glyph Forge validates the result before terminal presentation. Built-in
renderers can be composed by copying the supplied `RenderConfig` with a
built-in mode, as demonstrated by
[`examples/plugin_example.py`](../examples/plugin_example.py).

## Transform and exporter contracts

Transforms accept `TransformRequest(source, options)` and return the transformed
value. Exporters accept `ExportRequest(source, destination, options)` and return
an `ExportReceipt`. These contracts are available through `PluginRegistry` for
applications assembling custom pipelines:

```python
from glyph_forge.plugins import get_plugin_registry

registry = get_plugin_registry()
processed = registry.transform("my-effects/bloom", frame, options={"radius": 3})
receipt = registry.export("my-effects/archive", processed, "output.glyph")
```

Options are immutable mappings at the extension boundary. Exceptions from
extension code are wrapped with the qualified component name, while failures in
one plugin do not prevent other plugins from loading.

## Compatibility and diagnostics

`PLUGIN_API_VERSION` is currently `1`. A manifest must declare the exact API
version it implements; incompatible plugins fail before any component runs.
Normal application startup does not discover or import extensions. Commands:

```bash
# Metadata only; imports no plugin modules
glyph-forge plugins --json

# Load and validate one selected plugin
glyph-forge plugins inspect my-effects --json

# Probe every plugin independently; useful for support reports
glyph-forge plugins --probe
```

Plugin metadata also appears in `glyph-forge doctor --json`. Test extensions
against eco settings, terminal viewport limits, finite sources, cleanup after
errors, and supported Python versions. Keep heavyweight dependencies inside
the selected factory or component rather than importing them at module scope.

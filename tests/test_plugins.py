"""Tests for the versioned, lazy Glyph Forge extension boundary."""

from __future__ import annotations

import json
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from glyph_forge.cli import app
from glyph_forge.live.capture import IterableFrameSource, create_frame_source
from glyph_forge.live.renderers import FrameRenderer, RenderConfig
from glyph_forge.plugins import (
    PLUGIN_API_VERSION,
    ExportReceipt,
    PluginCompatibilityError,
    PluginConflictError,
    PluginContractError,
    PluginLoadError,
    PluginManifest,
    PluginRegistry,
    RenderOutput,
    SourceRequest,
    parse_component_reference,
)


class FakeEntryPoint:
    def __init__(
        self,
        name: str,
        exported: object,
        *,
        value: str = "example.plugin:manifest",
        distribution: str = "example-package",
        version: str = "2.4.0",
    ) -> None:
        self.name = name
        self.value = value
        self.dist = SimpleNamespace(
            metadata={"Name": distribution},
            version=version,
        )
        self.exported = exported
        self.loads = 0

    def load(self) -> object:
        self.loads += 1
        if isinstance(self.exported, BaseException):
            raise self.exported
        return self.exported


def manifest(**overrides: Any) -> PluginManifest:
    values: dict[str, Any] = {
        "name": "Example effects",
        "version": "1.3.0",
        "description": "A test plugin",
    }
    values.update(overrides)
    return PluginManifest(**values)


def test_component_reference_is_explicit_and_preserves_source_resource() -> None:
    renderer = parse_component_reference("plugin:demo/glow")
    source = parse_component_reference(
        "plugin:demo/network:https://example.test/live",
        allow_resource=True,
    )

    assert renderer.qualified == "demo/glow"
    assert renderer.resource == ""
    assert source.qualified == "demo/network"
    assert source.resource == "https://example.test/live"


@pytest.mark.parametrize(
    "value",
    ["plugin:missing-component", "plugin:/source", "plugin:pack/", "plugin:p/a/b"],
)
def test_component_reference_rejects_ambiguous_names(value: str) -> None:
    with pytest.raises(PluginContractError, match="must look like"):
        parse_component_reference(value)


def test_inventory_reads_metadata_without_importing_plugin() -> None:
    point = FakeEntryPoint("effects", manifest())
    registry = PluginRegistry(discoverer=lambda: [point])

    info = registry.inventory()

    assert point.loads == 0
    assert len(info) == 1
    assert info[0].identifier == "effects"
    assert info[0].state == "discovered"
    assert info[0].distribution == "example-package"
    assert info[0].version == "2.4.0"


def test_standard_library_entry_point_loads_runnable_example_manifest() -> None:
    point = EntryPoint(
        name="example",
        value="examples.plugin_example:plugin",
        group="glyph_forge.plugins",
    )
    registry = PluginRegistry(discoverer=lambda: [point])

    info = registry.info("example")

    assert info.state == "ready"
    assert info.sources == ("gradient",)
    assert info.renderers == ("outline",)
    assert info.transforms == ("invert",)
    assert info.exporters == ("text",)


def test_explicit_load_is_cached_and_exposes_components() -> None:
    point = FakeEntryPoint(
        "effects",
        lambda: manifest(
            sources={"noise": lambda request: object()},
            renderers={"glow": lambda request: object()},
            transforms={"blur": lambda request: request.source},
            exporters={"poster": lambda request: ExportReceipt(request.destination)},
        ),
    )
    registry = PluginRegistry(discoverer=lambda: [point])

    first = registry.info("effects")
    second = registry.info("EFFECTS")

    assert point.loads == 1
    assert first == second
    assert first.state == "ready"
    assert first.sources == ("noise",)
    assert first.renderers == ("glow",)
    assert first.transforms == ("blur",)
    assert first.exporters == ("poster",)


def test_probe_isolates_one_broken_plugin_from_healthy_plugins() -> None:
    broken = FakeEntryPoint("broken", ImportError("optional engine is missing"))
    healthy = FakeEntryPoint("healthy", manifest())
    registry = PluginRegistry(discoverer=lambda: [broken, healthy])

    results = {item.identifier: item for item in registry.inventory(load=True)}

    assert results["broken"].state == "error"
    assert "optional engine is missing" in (results["broken"].error or "")
    assert results["healthy"].state == "ready"
    assert broken.loads == healthy.loads == 1
    assert registry.inventory(load=False)[0].state == "error"


def test_plugin_system_exit_is_reported_without_exiting_host() -> None:
    point = FakeEntryPoint("exits", SystemExit(7))
    registry = PluginRegistry(discoverer=lambda: [point])

    with pytest.raises(PluginLoadError, match="SystemExit: 7"):
        registry.load("exits")


def test_duplicate_installed_identifiers_are_rejected_deterministically() -> None:
    registry = PluginRegistry(
        discoverer=lambda: [
            FakeEntryPoint("same", manifest(name="One")),
            FakeEntryPoint("SAME", manifest(name="Two")),
        ]
    )

    info = registry.inventory()

    assert info[0].state == "conflict"
    with pytest.raises(PluginConflictError, match="Multiple installed plugins"):
        registry.load("same")


def test_incompatible_api_is_rejected_before_component_execution() -> None:
    plugin = manifest(api_version=PLUGIN_API_VERSION + 1)
    registry = PluginRegistry(discoverer=lambda: [FakeEntryPoint("future", plugin)])

    with pytest.raises(PluginCompatibilityError, match="supports API"):
        registry.load("future")


def test_manifest_rejects_invalid_or_non_callable_components() -> None:
    registry = PluginRegistry(discoverer=lambda: ())

    with pytest.raises(PluginContractError, match="component"):
        registry.register(
            "bad",
            manifest(renderers={"Not Valid!": lambda request: object()}),
        )
    with pytest.raises(PluginContractError, match="not callable"):
        registry.register(
            "bad",
            manifest(exporters={"file": 42}),  # type: ignore[dict-item]
        )


def test_manual_registration_requires_explicit_replacement() -> None:
    registry = PluginRegistry(discoverer=lambda: ())
    registry.register("demo", manifest(name="First"))

    with pytest.raises(PluginConflictError, match="already registered"):
        registry.register("demo", manifest(name="Second"))

    registry.register("demo", manifest(name="Second"), replace=True)
    assert registry.info("demo").name == "Second"


def test_plugin_source_uses_live_pipeline_and_receives_capture_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[SourceRequest] = []

    def source_factory(request: SourceRequest) -> IterableFrameSource:
        requests.append(request)
        return IterableFrameSource(
            [np.full((2, 3, 3), 17, dtype=np.uint8)],
            name=f"generated:{request.resource}",
        )

    registry = PluginRegistry(discoverer=lambda: ())
    registry.register(
        "generators",
        manifest(sources={"solid": source_factory}),
    )
    import glyph_forge.plugins.registry as registry_module

    monkeypatch.setattr(registry_module, "_registry", registry)

    source = create_frame_source(
        "plugin:generators/solid:blue:fast",
        width=320,
        height=180,
        fps=24,
        loop=True,
    )
    frame = source.read()

    assert source.name == "generated:blue:fast"
    assert frame is not None and frame.shape == (2, 3, 3)
    assert requests == [SourceRequest("blue:fast", 320, 180, 24, True)]


def test_plugin_renderer_reuses_frame_renderer_and_viewport_contract() -> None:
    class TinyRenderer:
        def render(
            self,
            frame: np.ndarray,
            *,
            max_width: int | None = None,
            max_height: int | None = None,
        ) -> RenderOutput:
            width = min(3, max_width or 3)
            height = min(2, max_height or 2)
            assert frame.flags.c_contiguous
            return RenderOutput("\n".join(["#" * width] * height), width, height)

    registry = PluginRegistry(discoverer=lambda: ())
    registry.register(
        "effects",
        manifest(renderers={"tiny": lambda request: TinyRenderer()}),
    )
    renderer = FrameRenderer(
        RenderConfig(width=80, mode="plugin:effects/tiny"),
        plugin_registry=registry,
    )

    result = renderer.render(
        np.zeros((5, 7, 4), dtype=np.float32),
        max_width=2,
        max_height=1,
    )

    assert result.text == "##"
    assert (result.width, result.height) == (2, 1)
    assert result.mode.value == "plugin:effects/tiny"


def test_plugin_renderer_output_is_validated_at_trust_boundary() -> None:
    class BadRenderer:
        def render(self, frame: Any, **options: Any) -> RenderOutput:
            return RenderOutput("one row", 7, 2)

    registry = PluginRegistry(discoverer=lambda: ())
    registry.register(
        "bad",
        manifest(renderers={"rows": lambda request: BadRenderer()}),
    )
    renderer = FrameRenderer(
        RenderConfig(mode="plugin:bad/rows"),
        plugin_registry=registry,
    )

    with pytest.raises(PluginContractError, match="row count"):
        renderer.render(np.zeros((1, 1, 3), dtype=np.uint8))

    class BadDimensions:
        def render(self, frame: Any, **options: Any) -> RenderOutput:
            return RenderOutput("x", 1.5, 1)  # type: ignore[arg-type]

    registry.register(
        "bad-dimensions",
        manifest(renderers={"float": lambda request: BadDimensions()}),
    )
    invalid = FrameRenderer(
        RenderConfig(mode="plugin:bad-dimensions/float"),
        plugin_registry=registry,
    )
    with pytest.raises(PluginContractError, match="invalid dimensions"):
        invalid.render(np.zeros((1, 1, 3), dtype=np.uint8))


def test_transform_and_exporter_invocation_are_typed_and_isolated(
    tmp_path: Path,
) -> None:
    def uppercase(request: Any) -> str:
        return str(request.source).upper() + str(request.options["suffix"])

    def export(request: Any) -> ExportReceipt:
        request.destination.write_text(str(request.source), encoding="utf-8")
        return ExportReceipt(
            request.destination,
            "text/plain",
            {"characters": len(str(request.source))},
        )

    registry = PluginRegistry(discoverer=lambda: ())
    registry.register(
        "tools",
        manifest(transforms={"upper": uppercase}, exporters={"text": export}),
    )
    transformed = registry.transform(
        "tools/upper",
        "hello",
        options={"suffix": "!"},
    )
    destination = tmp_path / "result.txt"
    receipt = registry.export("tools/text", transformed, destination)

    assert transformed == "HELLO!"
    assert destination.read_text(encoding="utf-8") == "HELLO!"
    assert receipt.output == destination
    assert receipt.media_type == "text/plain"
    assert receipt.metadata["characters"] == 6


def test_cli_plugin_inventory_is_machine_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = PluginRegistry(discoverer=lambda: ())
    registry.register("demo", manifest(renderers={"tiny": lambda request: object()}))
    import glyph_forge.plugins.registry as registry_module

    monkeypatch.setattr(registry_module, "_registry", registry)

    result = CliRunner().invoke(app, ["plugins", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["api_version"] == PLUGIN_API_VERSION
    assert payload["plugins"][0]["identifier"] == "demo"
    assert payload["plugins"][0]["renderers"] == ["tiny"]


def test_cli_inspect_and_doctor_report_validated_plugin_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = PluginRegistry(discoverer=lambda: ())
    registry.register(
        "demo",
        manifest(
            sources={"frame": lambda request: object()},
            renderers={"tiny": lambda request: object()},
        ),
    )
    import glyph_forge.plugins.registry as registry_module

    monkeypatch.setattr(registry_module, "_registry", registry)
    runner = CliRunner()

    inspected = runner.invoke(app, ["plugins", "inspect", "demo", "--json"])
    diagnosed = runner.invoke(app, ["doctor", "--json"])

    assert inspected.exit_code == 0
    assert json.loads(inspected.stdout)["sources"] == ["frame"]
    assert diagnosed.exit_code == 0
    plugins = json.loads(diagnosed.stdout)["plugins"]
    assert plugins["api_version"] == PLUGIN_API_VERSION
    assert plugins["installed"][0]["identifier"] == "demo"


def test_top_level_stream_combines_plugin_source_and_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyRenderer:
        def render(self, frame: Any, **options: Any) -> RenderOutput:
            return RenderOutput("@@", 2, 1)

    registry = PluginRegistry(discoverer=lambda: ())
    registry.register(
        "demo",
        manifest(
            sources={
                "frame": lambda request: IterableFrameSource(
                    [np.zeros((2, 2, 3), dtype=np.uint8)],
                    name="plugin-frame",
                )
            },
            renderers={"tiny": lambda request: TinyRenderer()},
        ),
    )
    import glyph_forge.plugins.registry as registry_module

    monkeypatch.setattr(registry_module, "_registry", registry)

    result = CliRunner().invoke(
        app,
        [
            "stream",
            "plugin:demo/frame",
            "--mode",
            "plugin:demo/tiny",
            "--frames",
            "1",
            "--fps",
            "120",
            "--no-fit",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "@@" in result.output
    assert "1 displayed" in result.output


def test_environment_switch_disables_external_discovery_but_not_manual_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = FakeEntryPoint("external", manifest())
    monkeypatch.setenv("GLYPH_FORGE_DISABLE_PLUGINS", "true")
    registry = PluginRegistry(discoverer=lambda: [point])

    assert registry.inventory() == ()
    assert point.loads == 0
    registry.register("embedded", manifest())
    assert registry.info("embedded").state == "ready"

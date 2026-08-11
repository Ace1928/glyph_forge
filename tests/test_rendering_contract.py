"""Contract, fidelity, and persistence tests for the canonical still pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from glyph_forge.contracts import (
    RENDER_CONTRACT_VERSION,
    Alignment,
    FitMode,
    RenderContractError,
    RenderFormat,
    RenderRequest,
    SourceLoadError,
)
from glyph_forge.rendering import format_for_path, render_image, truecolor_to_html

_CONFORMANCE = json.loads(
    (Path(__file__).parent / "fixtures" / "render-contract-v1.json").read_text(
        encoding="utf-8"
    )
)


def neutral_request(**updates: object) -> RenderRequest:
    values: dict[str, Any] = {
        "width": 4,
        "height": 2,
        "charset": " .#",
        "brightness": 1.0,
        "contrast": 1.0,
        "resample": "nearest",
    }
    values.update(updates)
    return RenderRequest(**values)


def test_request_round_trips_as_a_versioned_json_contract() -> None:
    request = neutral_request(
        output_format="svg",
        output_width=640,
        output_height=360,
        fit="cover",
        alignment="top-right",
    )

    encoded = json.loads(json.dumps(request.to_dict()))
    restored = RenderRequest.from_dict(encoded)

    assert restored == request
    assert restored.contract_version == RENDER_CONTRACT_VERSION
    assert restored.render_format is RenderFormat.SVG
    assert restored.fit_mode is FitMode.COVER
    assert restored.alignment_mode is Alignment.TOP_RIGHT


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"threshold": "128"}, "threshold must be an integer"),
        ({"edge_threshold": 1.5}, "edge_threshold must be an integer"),
        ({"cell_aspect": "wide"}, "cell_aspect must be a number"),
        ({"resample": 7}, "resample must be a string"),
        ({"dither": "yes"}, "dither must be true or false"),
        ({"background": None}, "background must be a non-empty"),
    ],
)
def test_request_rejects_malformed_serialized_types(
    updates: dict[str, object],
    message: str,
) -> None:
    values = RenderRequest().to_dict() | updates

    with pytest.raises(RenderContractError, match=message):
        RenderRequest.from_dict(values)


def test_request_wraps_unknown_serialized_fields() -> None:
    with pytest.raises(RenderContractError, match="Malformed serialized"):
        RenderRequest.from_dict({"surprise": True})


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"contract_version": 99}, "contract version"),
        ({"width": 0}, "width"),
        ({"charset": ""}, "charset"),
        ({"brightness": float("nan")}, "finite"),
        ({"output_format": "jpeg"}, "output format"),
        ({"mode": "pixels"}, "render mode"),
        ({"edge_algorithm": "magic"}, "edge algorithm"),
        ({"output_width": 100}, "PNG or SVG"),
        ({"output_format": "html", "mode": "braille"}, "glyph mode"),
        ({"output_format": "truecolor", "style": "boxed"}, "styles"),
    ],
)
def test_request_rejects_ambiguous_or_unsafe_values(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(RenderContractError, match=message):
        neutral_request(**updates)


def test_neutral_glyph_mapping_is_a_stable_golden_contract() -> None:
    row = np.asarray([0, 64, 128, 255], dtype=np.uint8)
    source = np.repeat(row[None, :, None], 3, axis=2)
    source = np.repeat(source, 2, axis=0)

    artifact = render_image(source, neutral_request())

    assert artifact.glyph_text == "  .#\n  .#"
    assert (artifact.columns, artifact.rows) == (4, 2)
    assert artifact.data == artifact.glyph_text
    assert artifact.media_type == "text/plain; charset=utf-8"
    assert artifact.metrics.source_width == 4
    assert artifact.metrics.source_height == 2
    assert artifact.metrics.output_bytes == len(artifact.data.encode("utf-8"))


def test_explicit_plugin_renderer_reference_is_contract_valid() -> None:
    request = neutral_request(mode="plugin:example/renderer")

    assert request.mode == "plugin:example/renderer"


def test_charset_typo_fails_before_source_decode() -> None:
    with pytest.raises(RenderContractError, match="did you mean detailed"):
        render_image("also-missing.png", neutral_request(charset="detaled"))


@pytest.mark.parametrize(
    "case",
    _CONFORMANCE["cases"],
    ids=[case["name"] for case in _CONFORMANCE["cases"]],
)
def test_native_renderer_matches_shared_cross_runtime_fixtures(
    case: dict[str, Any],
) -> None:
    sample = case["sample"]
    rgba = np.asarray(sample["rgba"], dtype=np.uint8).reshape(
        sample["height"],
        sample["width"],
        4,
    )

    artifact = render_image(rgba, RenderRequest.from_dict(case["request"]))

    assert artifact.glyph_text.splitlines() == case["expected_lines"]
    for red, green, blue in case.get("expected_rgb", []):
        assert f"\x1b[38;2;{red};{green};{blue}m" in artifact.data


@pytest.mark.parametrize("mode", ["glyph", "edge", "braille", "half-block", "quadrant"])
def test_every_native_mode_uses_the_same_artifact_contract(mode: str) -> None:
    source = np.arange(16 * 12 * 3, dtype=np.uint8).reshape(12, 16, 3)
    artifact = render_image(source, neutral_request(mode=mode))

    assert (artifact.columns, artifact.rows) == (4, 2)
    assert len(artifact.glyph_text.splitlines()) == 2
    assert artifact.request.mode == mode


def test_truecolor_html_is_escaped_and_contains_source_color() -> None:
    source = np.asarray([[[17, 34, 51]]], dtype=np.uint8)
    artifact = render_image(
        source,
        RenderRequest(
            width=1,
            height=1,
            charset="<",
            brightness=1.0,
            contrast=1.0,
            output_format="html",
        ),
    )

    assert artifact.glyph_text == "<"
    assert "color:#112233" in str(artifact.data)
    assert "&lt;" in str(artifact.data)
    assert "><</span>" not in str(artifact.data)


def test_html_converter_preserves_uncolored_text_safely() -> None:
    assert truecolor_to_html("<unsafe>&") == (
        "<pre style='line-height:1; letter-spacing:0'>&lt;unsafe&gt;&amp;</pre>"
    )


def test_ansi_suffix_preserves_an_explicit_256_colour_request() -> None:
    assert format_for_path("art.ansi", color="ansi256") is RenderFormat.ANSI256


@pytest.mark.parametrize("field", ["foreground", "background"])
def test_invalid_graphical_colors_fail_before_rendering(field: str) -> None:
    with pytest.raises(RenderContractError, match=f"Invalid {field} color"):
        render_image(
            np.zeros((1, 1, 3), dtype=np.uint8),
            neutral_request(**{field: "definitely-not-a-colour"}),
        )


@pytest.mark.parametrize("output_format", [RenderFormat.PNG, RenderFormat.SVG])
def test_graphical_artifacts_keep_pixels_independent_from_cells(
    output_format: RenderFormat,
) -> None:
    request = neutral_request(
        output_format=output_format,
        output_width=333,
        output_height=211,
        fit="contain",
        alignment="bottom-right",
    )
    artifact = render_image(np.zeros((8, 8, 3), dtype=np.uint8), request)

    assert (artifact.columns, artifact.rows) == (4, 2)
    assert (artifact.pixel_width, artifact.pixel_height) == (333, 211)
    if output_format is RenderFormat.PNG:
        assert isinstance(artifact.data, bytes)
        assert artifact.data.startswith(b"\x89PNG\r\n\x1a\n")
    else:
        assert isinstance(artifact.data, str)
        assert 'viewBox="0 0 333.00 211.00"' in artifact.data
        assert "translate(" in artifact.data


def test_transparent_source_is_composited_against_requested_background() -> None:
    source = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    artifact = render_image(
        source,
        RenderRequest(
            width=1,
            height=1,
            charset=" .",
            background="#000000",
            brightness=1.0,
            contrast=1.0,
        ),
    )

    assert artifact.glyph_text == " "


def test_artifact_save_is_atomic_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    artifact = render_image(np.zeros((2, 2, 3), dtype=np.uint8), neutral_request())
    destination = tmp_path / "nested" / "art.txt"

    returned = artifact.save(destination)

    assert returned == destination
    assert destination.read_text(encoding="utf-8") == artifact.glyph_text
    assert list(destination.parent.glob(".*.tmp")) == []


def test_invalid_source_raises_a_typed_failure() -> None:
    with pytest.raises(SourceLoadError, match="Could not load image"):
        render_image("definitely-missing.png", neutral_request())


def test_path_dimensions_are_bounded_before_pixel_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glyph_forge import rendering

    source = tmp_path / "bounded.png"
    Image.new("RGB", (2, 2)).save(source)
    monkeypatch.setattr(rendering, "MAX_SOURCE_PIXELS", 1)

    with pytest.raises(SourceLoadError, match="decode budget"):
        render_image(source, neutral_request())

"""Tests for compact historical command launchers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from glyph_forge.cli import app, bannerize, imagize


def test_imagize_legacy_short_options_reach_unified_image(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "output.txt"
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(source)

    status = imagize.main(
        [
            "-s",
            "blocks",
            "-c",
            "none",
            "-w",
            "8",
            "-h",
            "4",
            "--no-fit",
            "--no-preview",
            "-o",
            str(output),
            str(source),
        ]
    )

    assert status == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 4


def test_bannerize_launcher_and_helper_remain_available(capsys) -> None:
    status = bannerize.main(
        ["Hello", "--font", "small", "--style", "minimal", "--width", "40"]
    )

    assert status == 0
    assert capsys.readouterr().out.strip()
    assert bannerize.create_banner("Hi", font="small").strip()


def test_legacy_information_modes_need_no_source(capsys) -> None:
    assert imagize.main(["--preview-charset", "blocks"]) == 0
    assert "blocks" in capsys.readouterr().out
    assert bannerize.main(["--list-styles"]) == 0
    assert "minimal" in capsys.readouterr().out


def test_unified_hidden_compatibility_names_still_work() -> None:
    runner = CliRunner()

    image = runner.invoke(app, ["imagize", "--list-charsets"])
    text = runner.invoke(app, ["bannerize", "--list-styles"])

    assert image.exit_code == 0, image.output
    assert text.exit_code == 0, text.output

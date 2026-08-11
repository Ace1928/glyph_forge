"""Bounded cross-interface still batch queue tests."""

from __future__ import annotations

import threading
from pathlib import Path
from time import monotonic

import numpy as np
import pytest
from PIL import Image

from glyph_forge import batch
from glyph_forge.batch import (
    BatchError,
    BatchProgress,
    BatchRenderItem,
    CancellationToken,
    items_for_sources,
    render_batch,
)
from glyph_forge.contracts import RenderRequest


def image(path: Path, value: int = 128) -> None:
    Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(path)


def test_batch_renders_in_input_order_with_bounded_parallel_workers(
    tmp_path: Path,
) -> None:
    sources = [tmp_path / f"source-{index}.png" for index in range(4)]
    for index, source in enumerate(sources):
        image(source, index * 50)
    request = RenderRequest(width=4, height=2, brightness=1, contrast=1)
    items = items_for_sources(sources, tmp_path / "out", request)
    progress: list[BatchProgress] = []

    report = render_batch(items, workers=3, progress=progress.append)

    assert report.total_items == 4
    assert report.succeeded == 4
    assert report.failed == report.skipped == 0
    assert [item.index for item in report.results] == [0, 1, 2, 3]
    assert all(item.destination.is_file() for item in report.results)
    assert [item.completed for item in progress] == [1, 2, 3, 4]
    assert report.to_dict()["workers"] == 3


def test_batch_isolates_item_failures_and_keeps_successful_outputs(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.png"
    image(valid)
    request = RenderRequest(width=2, height=1)
    items = items_for_sources(
        [tmp_path / "missing.png", valid], tmp_path / "out", request
    )

    report = render_batch(items, workers=1)

    assert report.succeeded == 1
    assert report.failed == 1
    assert "SourceLoadError" in str(report.results[0].error)
    assert report.results[1].destination.is_file()


def test_batch_names_colliding_stems_without_overwriting(tmp_path: Path) -> None:
    first = tmp_path / "a" / "same.png"
    second = tmp_path / "b" / "same.jpg"
    first.parent.mkdir()
    second.parent.mkdir()
    image(first)
    image(second)

    items = items_for_sources(
        [first, second], tmp_path / "out", RenderRequest(output_format="svg")
    )

    assert [item.destination.name for item in items] == [
        "same.glyph.svg",
        "same-2.glyph.svg",
    ]


def test_batch_rejects_empty_duplicate_or_unbounded_queues(tmp_path: Path) -> None:
    with pytest.raises(BatchError, match="at least one"):
        render_batch([])
    source = tmp_path / "source.png"
    image(source)
    item = BatchRenderItem(source, tmp_path / "out.txt", RenderRequest())
    with pytest.raises(BatchError, match="unique"):
        render_batch([item, item])
    with pytest.raises(BatchError, match="workers"):
        render_batch([item], workers=0)
    monkey_items = [
        BatchRenderItem(source, tmp_path / f"{index}.txt", RenderRequest())
        for index in range(batch.MAX_BATCH_ITEMS + 1)
    ]
    with pytest.raises(BatchError, match="cannot exceed"):
        render_batch(monkey_items)


def test_cancellation_stops_new_submissions_with_only_worker_bound_in_flight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = CancellationToken()
    started: list[int] = []
    release = threading.Event()
    real_render = batch.render_image
    sources = [tmp_path / f"{index}.png" for index in range(8)]
    for source in sources:
        image(source)
    items = items_for_sources(sources, tmp_path / "out", RenderRequest(width=2))

    def controlled_render(*args, **kwargs):  # type: ignore[no-untyped-def]
        started.append(len(started))
        if len(started) == 2:
            token.cancel()
            release.set()
        release.wait(timeout=1)
        return real_render(*args, **kwargs)

    monkeypatch.setattr(batch, "render_image", controlled_render)
    before = monotonic()

    report = render_batch(items, workers=2, cancellation=token)

    assert report.cancelled
    assert len(started) <= 2
    assert report.skipped >= 6
    assert monotonic() - before < 2

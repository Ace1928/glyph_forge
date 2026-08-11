"""Bounded still-image batch rendering shared by application interfaces."""

from __future__ import annotations

import os
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

from .contracts import RenderArtifact, RenderRequest
from .rendering import render_image

MAX_BATCH_ITEMS = 1000
MAX_BATCH_WORKERS = 64


class BatchError(Exception):
    """A batch request is invalid or cannot be completed."""


class BatchCancelled(BatchError):
    """A batch stopped in response to an explicit cancellation request."""


@dataclass(frozen=True, slots=True)
class BatchRenderItem:
    """One source, destination, and immutable render request."""

    source: Path
    destination: Path
    request: RenderRequest

    def __post_init__(self) -> None:
        source = Path(self.source).expanduser()
        destination = Path(self.destination).expanduser()
        if source.resolve() == destination.resolve():
            raise BatchError("batch source and destination must be different")
        if not isinstance(self.request, RenderRequest):
            raise BatchError("batch request must be a RenderRequest")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "destination", destination)


@dataclass(frozen=True, slots=True)
class BatchItemResult:
    """Outcome for one item; failures do not discard successful siblings."""

    index: int
    source: Path
    destination: Path
    artifact: RenderArtifact | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.artifact is not None and self.error is None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "index": self.index,
            "source": str(self.source),
            "destination": str(self.destination),
            "succeeded": self.succeeded,
            "error": self.error,
        }
        if self.artifact is not None:
            result.update(
                {
                    "columns": self.artifact.columns,
                    "rows": self.artifact.rows,
                    "pixel_width": self.artifact.pixel_width,
                    "pixel_height": self.artifact.pixel_height,
                    "media_type": self.artifact.media_type,
                    "metrics": self.artifact.metrics.to_dict(),
                }
            )
        return result


@dataclass(frozen=True, slots=True)
class BatchProgress:
    """Monotonic progress snapshot emitted after each completed item."""

    completed: int
    total: int
    succeeded: int
    failed: int
    elapsed: float

    @property
    def fraction(self) -> float:
        return self.completed / self.total


@dataclass(frozen=True, slots=True)
class BatchRenderReport:
    """Ordered outcomes and aggregate performance for a completed batch."""

    results: tuple[BatchItemResult, ...]
    total_items: int
    elapsed: float
    workers: int
    cancelled: bool = False

    @property
    def succeeded(self) -> int:
        return sum(item.succeeded for item in self.results)

    @property
    def failed(self) -> int:
        return len(self.results) - self.succeeded

    @property
    def skipped(self) -> int:
        return self.total_items - len(self.results)

    @property
    def output_bytes(self) -> int:
        return sum(
            item.artifact.byte_size
            for item in self.results
            if item.artifact is not None
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "items": self.total_items,
            "completed": len(self.results),
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "cancelled": self.cancelled,
            "elapsed_seconds": self.elapsed,
            "workers": self.workers,
            "output_bytes": self.output_bytes,
            "results": [item.to_dict() for item in self.results],
        }


class CancellationToken:
    """Thread-safe cooperative cancellation signal reusable by UIs."""

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise BatchCancelled("batch was cancelled")


ProgressCallback = Callable[[BatchProgress], None]


def _render_one(index: int, item: BatchRenderItem) -> BatchItemResult:
    try:
        artifact = render_image(
            item.source,
            item.request,
            destination=item.destination,
        )
    except Exception as exc:
        return BatchItemResult(
            index,
            item.source,
            item.destination,
            error=f"{type(exc).__name__}: {exc}",
        )
    return BatchItemResult(
        index,
        item.source,
        item.destination,
        artifact=artifact,
    )


def _validated_items(items: Iterable[BatchRenderItem]) -> tuple[BatchRenderItem, ...]:
    selected = tuple(items)
    if not selected:
        raise BatchError("batch requires at least one item")
    if len(selected) > MAX_BATCH_ITEMS:
        raise BatchError(f"batch cannot exceed {MAX_BATCH_ITEMS} items")
    if not all(isinstance(item, BatchRenderItem) for item in selected):
        raise BatchError("every batch entry must be a BatchRenderItem")
    destinations = [item.destination.resolve() for item in selected]
    if len(set(destinations)) != len(destinations):
        raise BatchError("batch destinations must be unique")
    return selected


class _BatchRunner:
    """Small scheduler keeping queue bounds and progress bookkeeping explicit."""

    def __init__(
        self,
        items: tuple[BatchRenderItem, ...],
        workers: int,
        token: CancellationToken,
        progress: ProgressCallback | None,
        fail_fast: bool,
        started: float,
    ) -> None:
        self.items = items
        self.workers = workers
        self.token = token
        self.progress = progress
        self.fail_fast = fail_fast
        self.started = started
        self.completed: dict[int, BatchItemResult] = {}
        self.pending: dict[Future[BatchItemResult], int] = {}
        self.next_index = 0
        self.succeeded = 0
        self.stopped = False

    @property
    def can_submit(self) -> bool:
        return self.next_index < len(self.items) and not self.stopped

    def submit_available(self, executor: ThreadPoolExecutor) -> None:
        while (
            self.can_submit
            and not self.token.cancelled
            and len(self.pending) < self.workers
        ):
            future = executor.submit(
                _render_one,
                self.next_index,
                self.items[self.next_index],
            )
            self.pending[future] = self.next_index
            self.next_index += 1

    def request_stop(self) -> None:
        self.stopped = True
        for future in self.pending:
            future.cancel()

    def collect(self, finished: set[Future[BatchItemResult]]) -> None:
        for future in finished:
            index = self.pending.pop(future)
            if future.cancelled():
                continue
            result = future.result()
            self.completed[index] = result
            self.succeeded += result.succeeded
            if self.fail_fast and not result.succeeded:
                self.stopped = True
            self.report_progress()

    def report_progress(self) -> None:
        if self.progress is None:
            return
        self.progress(
            BatchProgress(
                completed=len(self.completed),
                total=len(self.items),
                succeeded=self.succeeded,
                failed=len(self.completed) - self.succeeded,
                elapsed=perf_counter() - self.started,
            )
        )

    def run(self, executor: ThreadPoolExecutor) -> dict[int, BatchItemResult]:
        while self.pending or self.can_submit:
            self.submit_available(executor)
            if self.token.cancelled:
                self.request_stop()
            if not self.pending:
                break
            finished, _ = wait(tuple(self.pending), return_when=FIRST_COMPLETED)
            self.collect(finished)
        return self.completed


def render_batch(
    items: Iterable[BatchRenderItem],
    *,
    workers: int = 1,
    cancellation: CancellationToken | None = None,
    progress: ProgressCallback | None = None,
    fail_fast: bool = False,
) -> BatchRenderReport:
    """Render a bounded queue with at most one in-flight item per worker.

    Results retain input order even when workers complete out of order.  Item
    failures are isolated by default; ``fail_fast`` stops scheduling new work
    after the first failure.
    """

    selected = _validated_items(items)
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise BatchError("workers must be an integer")
    if not 1 <= workers <= MAX_BATCH_WORKERS:
        raise BatchError(f"workers must be between 1 and {MAX_BATCH_WORKERS}")
    active_workers = min(workers, len(selected))
    token = cancellation or CancellationToken()
    started = perf_counter()
    executor = ThreadPoolExecutor(
        max_workers=active_workers,
        thread_name_prefix="glyph-forge-batch",
    )
    runner = _BatchRunner(
        selected,
        active_workers,
        token,
        progress,
        fail_fast,
        started,
    )
    try:
        completed = runner.run(executor)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    results = tuple(completed[index] for index in sorted(completed))
    return BatchRenderReport(
        results=results,
        total_items=len(selected),
        elapsed=perf_counter() - started,
        workers=active_workers,
        cancelled=token.cancelled,
    )


def output_suffix(request: RenderRequest) -> str:
    """Return the canonical filename suffix for a request's output format."""

    return {
        "text": ".txt",
        "ansi256": ".ansi",
        "truecolor": ".ansi",
        "html": ".html",
        "png": ".png",
        "svg": ".svg",
    }[request.render_format.value]


def items_for_sources(
    sources: Iterable[str | os.PathLike[str]],
    output_directory: str | os.PathLike[str],
    request: RenderRequest,
) -> tuple[BatchRenderItem, ...]:
    """Build collision-safe batch items with predictable output names."""

    output_dir = Path(output_directory).expanduser()
    suffix = output_suffix(request)
    result: list[BatchRenderItem] = []
    names: set[str] = set()
    for source_value in sources:
        source = Path(source_value).expanduser()
        stem = source.stem or "glyph"
        name = f"{stem}.glyph{suffix}"
        counter = 2
        while name.casefold() in names:
            name = f"{stem}-{counter}.glyph{suffix}"
            counter += 1
        names.add(name.casefold())
        result.append(BatchRenderItem(source, output_dir / name, request))
    return _validated_items(result)


__all__ = [
    "BatchCancelled",
    "BatchError",
    "BatchItemResult",
    "BatchProgress",
    "BatchRenderItem",
    "BatchRenderReport",
    "CancellationToken",
    "MAX_BATCH_ITEMS",
    "MAX_BATCH_WORKERS",
    "ProgressCallback",
    "items_for_sources",
    "output_suffix",
    "render_batch",
]

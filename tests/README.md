# Test suite

The suite covers the public API, image pipeline, renderers, terminal adaptation,
configuration, CLI boundaries, metadata, streaming components, optional TUI,
and integration workflows.

```bash
python -m pytest -q --tb=short
```

Tests use generated media and temporary directories. Optional tests skip with a
clear reason when their feature extra is not installed. CI runs both the core
suite and a complete optional-feature suite.

Benchmarks live in `tests/benchmarks` and require `pytest-benchmark`.

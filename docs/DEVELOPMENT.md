# Development guide

## Environment

```bash
git clone https://github.com/Ace1928/glyph_forge.git
cd glyph_forge
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install optional stacks only when working on them:

```bash
python -m pip install -e ".[dev,streaming,tui]"
```

The repository does not require Eidosian Forge, sibling source trees, or a
modified `PYTHONPATH`.

## Local release gates

Run these from the repository root:

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy --follow-imports=skip \
  src/glyph_forge/_version.py \
  src/glyph_forge/__init__.py \
  src/glyph_forge/exceptions.py \
  src/glyph_forge/terminal.py \
  src/glyph_forge/services/image_to_glyph.py \
  src/glyph_forge/streaming/naming.py \
  src/glyph_forge/streaming/core/buffer.py \
  src/glyph_forge/streaming/core/config.py \
  src/glyph_forge/streaming/core/renderer.py
python -m pytest -q --tb=short
python -m build
python -m twine check dist/*
```

The type gate intentionally covers the stable rendering surfaces. Legacy
capture adapters are exercised by tests while their platform-specific typing is
progressively tightened.

## Test layers

- Core tests use only package and development dependencies.
- Optional streaming tests require OpenCV and related extras.
- TUI tests require Textual.
- Benchmark tests require `pytest-benchmark`.
- Integration tests use temporary directories and generated fixtures; they do
  not download media.

Useful commands:

```bash
python -m pytest tests/test_api.py
python -m pytest -m integration
python -m pytest tests/benchmarks --benchmark-only
python scripts/benchmark.py --help
```

## Coding standards

- Support Python 3.10 through 3.13.
- Keep package import free of filesystem, logging, subprocess, and signal side
  effects.
- Use specific exceptions and preserve their causes with `raise ... from ...`.
- Keep optional imports inside the feature that needs them.
- Validate dimensions, paths, formats, and destructive intent at public
  boundaries.
- Use `pathlib`, UTF-8, atomic writes, and cross-platform standard-library APIs.
- Add a deterministic test for each behavior change.
- Document the supported contract, not internal aspirations.

## CI

GitHub Actions runs formatting, linting, stable-surface typing, Python 3.10-3.13
tests, Linux/macOS/Windows smoke tests, optional-feature tests, package metadata
validation, wheel installation, and a dependency vulnerability audit.

CI jobs use least-privilege read permissions, explicit timeouts, and concurrency
cancellation. Builds are uploaded as short-lived workflow artifacts.

## Releases

1. Move user-visible entries from `Unreleased` into a versioned changelog
   section.
2. Update both `pyproject.toml` and `src/glyph_forge/_version.py`.
3. Run every local release gate in a clean environment.
4. Install the built wheel in a new virtual environment and exercise the CLI.
5. Tag the reviewed commit with `vX.Y.Z`.
6. Publish through a protected, trusted-publishing workflow when configured.

Do not publish directly from an unreviewed working tree.

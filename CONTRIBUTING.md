# Contributing to Glyph Forge

Thanks for helping make Glyph Forge faster, friendlier, and more portable.

## Set up

```bash
git clone https://github.com/Ace1928/glyph_forge.git
cd glyph_forge
python -m venv .venv

# Linux/macOS
. .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --editable ".[dev,tui]"
python -m pytest
```

The repository has one long-lived branch: `main`. Maintainers land tested,
reviewable checkpoints there. External contributors can use a short-lived fork
branch for a pull request; delete it after merge rather than creating another
permanent repository branch.

## Before changing code

- Search existing issues and open one for a substantial feature or behavior
  change.
- Keep optional features behind lazy imports and an existing or new extra.
- Reuse the capture, render, runtime, and presentation protocols instead of
  adding a parallel command-specific engine.
- Use the [extension API](docs/extensions.md) for optional third-party behavior;
  do not add package scanning or eager plugin imports.
- Preserve documented compatibility unless a breaking change is discussed and
  scheduled.
- Measure performance claims with `glyph-forge benchmark` or a focused,
  reproducible benchmark.

## Quality gate

Run the same checks as CI:

```bash
ruff format src tests examples tools
ruff check src tests examples tools
python -m mypy src/glyph_forge
python -m pytest
python -m pytest --cov=glyph_forge --cov-fail-under=70
SOURCE_DATE_EPOCH=$(git log -1 --format=%ct) python -m build
python -m twine check dist/*
```

New behavior needs focused tests. Tests should be deterministic, independent,
fast by default, and should not require a physical webcam, display, network
connection, or FFmpeg process unless explicitly marked as an integration smoke
test. Mock an optional boundary rather than its internal rendering logic.

Release maintainers can dry-run the exact portable workflow from GitHub's
Actions tab. A `vVERSION` tag is accepted only when it exactly matches
`pyproject.toml`; only tag-triggered runs publish a GitHub release and signed
provenance. Native Windows/macOS certificate signing must be handled by an
authorized maintainer and must never place signing material in the repository.

## Design expectations

- Validate inputs at public boundaries and return actionable errors.
- Keep the hot path proportional to output cells whenever possible.
- Bound queues and temporary storage in live or batch workflows.
- Use platform-neutral interfaces, then isolate OS-specific implementations.
- Add type annotations to maintained public and internal functions.
- Document permission, privacy, portability, and fallback behavior accurately.
- Remove superseded code once compatibility is covered by a small adapter.

## Commits and pull requests

Use a short imperative subject, for example:

```text
Add Wayland portal capture adapter
Fix ANSI reset after half-block frames
Document virtual display permissions
```

A pull request should explain the user outcome, important design decisions,
tests run, platforms exercised, and any performance or compatibility impact.
Include terminal output, screenshots, or short captures when a visual change is
hard to assess from code.

By contributing, you agree that your work is licensed under the repository's
[MIT License](LICENSE) and to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

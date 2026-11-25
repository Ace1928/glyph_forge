# Contributing

Thank you for improving Glyph Forge. Small, focused changes with reproducible
tests are the easiest to review and maintain.

## Before coding

- Search existing issues and pull requests.
- Open an issue for a large feature, public API change, new required dependency,
  or format change before investing substantial work.
- Keep unrelated cleanup out of a focused fix.
- Never commit private media, credentials, browser cookies, or generated caches.

Security reports follow [SECURITY.md](SECURITY.md), not the public issue tracker.

## Setup

```bash
git clone https://github.com/Ace1928/glyph_forge.git
cd glyph_forge
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

Use a branch based on `master`, the repository's default branch.

## Required checks

```bash
python -m ruff format .
python -m ruff check .
python -m pytest -q --tb=short
```

Changes to streaming or the TUI should also be tested with:

```bash
python -m pip install -e ".[dev,streaming,tui]"
python -m pytest -q --tb=short
```

The full type and packaging commands are listed in
[the development guide](docs/DEVELOPMENT.md).

## Change requirements

- Validate user input before filesystem, network, capture, or subprocess work.
- Keep core features independent from optional dependencies.
- Preserve ASCII, redirected-output, `NO_COLOR`, and narrow-terminal behavior.
- Use package exception types at public boundaries and preserve causes.
- Keep writes atomic where partial output would be harmful.
- Add fast, deterministic tests using temporary directories and generated data.
- Update API, compatibility, and changelog documentation with behavior changes.
- Add a deprecation path instead of breaking documented 1.x interfaces.

Run benchmarks for hot-path changes, but do not reject a correctness improvement
solely because of noise in a single benchmark run.

## Pull requests

A useful pull request includes:

- the problem and intended behavior;
- the implementation approach and important tradeoffs;
- exact verification commands and results;
- screenshots or sample output only when visual behavior changed;
- compatibility, migration, and security notes where relevant.

Keep commits understandable. Conventional prefixes such as `feat:`, `fix:`,
`perf:`, `docs:`, and `test:` are welcome but not mandatory.

By contributing, you agree that your contribution is licensed under the MIT
License and that project interactions follow the [Code of Conduct](CODE_OF_CONDUCT.md).

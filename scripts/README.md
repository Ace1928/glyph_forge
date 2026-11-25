# Maintainer scripts

`benchmark.py` is the only supported repository script. It measures the
streaming renderer with generated frames and has no network or media-file
dependency.

```bash
python scripts/benchmark.py --help
```

Normal users should use the `glyph-forge` CLI. Tests, linting, builds, and wheel
validation use their standard `python -m ...` commands documented in
`docs/DEVELOPMENT.md` rather than wrapper scripts.

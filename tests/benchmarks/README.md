# Rendering benchmarks

These focused pytest benchmarks measure terminal detection, core image
conversion, and the optional streaming renderer. They use generated,
deterministic inputs and never invoke network, browser, capture, audio, or
subprocess operations.

```bash
python -m pytest tests/benchmarks --benchmark-only
```

The OpenCV benchmark skips when the streaming extra is absent; the other two
benchmarks run with the core installation.

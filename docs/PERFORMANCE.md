# Performance and benchmarking

Glyph Forge concentrates optimization in operations performed for every pixel
or frame. It does not promise a fixed frame rate because throughput depends on
source decoding, terminal size, color mode, font rendering, CPU, and terminal
emulator.

## Hot-path design

- Image pixels are processed in NumPy arrays.
- Luminance-to-glyph selection uses a 256-entry lookup table.
- sRGB linearization uses a 256-entry float lookup table.
- ANSI 256 matching uses a cached 32 by 32 by 32 color cube.
- Repeated color codes are run-length grouped before string assembly.
- Streaming acquisition and display are separated by an adaptive buffer.
- Cache archives are compressed, validated, non-executable NPZ files.

Plain output is fastest. ANSI 256 adds palette lookup and escape-sequence
assembly. Truecolor writes the most bytes and is commonly limited by terminal
rendering rather than conversion.

## Reproducible benchmark

Install the streaming and development extras, then run:

```bash
python -m pip install -e ".[dev,streaming]"
python scripts/benchmark.py --iterations 100 --mode ansi256
```

The benchmark generates deterministic in-memory frames, warms the renderer,
uses `time.perf_counter`, and reports median, mean, percentile, and estimated
frames per second. Record the following with any published result:

- Glyph Forge and Python versions;
- operating system and CPU;
- OpenCV and NumPy versions;
- source and terminal dimensions;
- glyph and color modes;
- warm-up and measured iteration counts.

Do not compare results that changed more than one of these variables.

## Profiling guidance

Use `cProfile` for a complete conversion and a sampling profiler for real-time
streaming. Exclude source download time when measuring rendering, but include
decode time when measuring end-to-end playback.

Performance changes must preserve output tests and renderer quality fixtures.
A faster path that changes glyph orientation, color ordering, ANSI reset
boundaries, or aspect ratio is a behavior change, not a pure optimization.

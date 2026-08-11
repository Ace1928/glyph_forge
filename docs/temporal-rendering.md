# Temporal rendering contract v1

Glyph Forge resolves every time-based render onto one explicit integer-frame
timeline before decoding or encoding. The contract is backend-free: importing
it does not load OpenCV, FFmpeg, a display server, or a UI toolkit.

## Minimal API

```python
from glyph_forge import FrameRate, TemporalRenderRequest

request = TemporalRenderRequest(
    start=2.5,
    duration=4.25,
    frame_rate="30000/1001",
    audio="preserve",
    rounding="nearest",
)
timeline = request.resolve(FrameRate(24), source_frames=12_000)

print(timeline.start_frame, timeline.frame_count)
print(timeline.ffmpeg_start, timeline.ffmpeg_duration)
print(timeline.to_dict())
```

`TemporalRenderRequest.to_dict()` returns a JSON-compatible object containing
`contract_version: 1`. `from_dict()` rejects unknown fields, unsupported
versions, booleans masquerading as numbers, non-finite or negative times,
invalid frame rates, and unknown policies with `TemporalContractError`.

## Exact cadence and deterministic slices

Frame rates are reduced rational values, not binary floating-point estimates.
Common fractional cadences therefore remain exact:

```python
FrameRate.parse("30000/1001")  # 29.97002997… fps
FrameRate.parse("24000/1001")  # 23.97602397… fps
```

Times are aligned once using `nearest`, `floor`, or `ceil`. The resolved
timeline then owns the aligned start frame, optional frame count, end frame,
encoded duration, and FFmpeg-safe values. Decode seeking, audio seeking,
progress totals, and result metrics all consume those same boundaries. A
requested duration is clamped at the known source end and a positive slice is
never rounded down to zero frames.

The default frame rate is the decoded source cadence. Set an explicit rate to
repair unreliable source metadata or deliberately re-time a decoded sequence.
Version 1 normalizes a variable-frame-rate stream to one constant cadence;
preserving arbitrary per-frame presentation timestamps requires a future
contract version.

## Audio policy

`audio="preserve"` aligns optional source audio to the selected starting frame
and bounds it to the encoded frame duration. `audio="discard"` creates a
silent artifact and avoids opening the source as a second FFmpeg input. An
input without audio remains valid because the source audio mapping is optional.

The command-line equivalents are:

```bash
# Preserve an exact NTSC cadence and align the requested slice to nearest frame
glyph-forge video clip.mov output.mp4 --frame-rate 30000/1001 \
  --start 2.5 --duration 4.25 --frame-rounding nearest

# Explicit silent export
glyph-forge video clip.mov output.mp4 --no-audio
```

`VideoExportConfig.temporal_request()` is the compatibility bridge for the
streamed exporter. Existing `start`, `duration`, and floating `fps` result
properties remain available; JSON metrics additionally include the complete
resolved timeline so automation can reproduce and audit an export exactly.

## Contract limits and evolution

- Frame rates must be positive and no greater than 1,000 frames per second;
  rational denominators are bounded to 1,000,000.
- Start and duration are finite and bounded to 366 days.
- Source and resolved frame counts are non-negative integers.
- Contract documents use strict fields; compatible additions require a new
  contract version when an older reader could interpret them ambiguously.

The temporal contract complements the independent
[still rendering contract](rendering.md). Visual settings control glyph detail
and output pixels; temporal settings control when frames occur and how audio is
aligned. Neither layer silently rewrites the other.

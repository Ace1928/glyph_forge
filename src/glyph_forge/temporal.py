"""Versioned, backend-free contracts for time-based rendering.

The temporal contract keeps frame-rate and slice decisions independent from
OpenCV, FFmpeg, capture devices, and UI code.  Consumers resolve a portable
request once, then use the resulting integer frame boundaries for decoding,
audio seeking, progress, and metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, Mapping, cast

TEMPORAL_CONTRACT_VERSION = 1
MAX_FRAME_RATE = 1_000
MAX_FRAME_RATE_DENOMINATOR = 1_000_000
MAX_TIMELINE_SECONDS = 366 * 24 * 60 * 60


class TemporalContractError(ValueError):
    """A temporal request or resolved timeline is malformed."""


class AudioPolicy(str, Enum):
    """How source audio participates in a rendered temporal artifact."""

    PRESERVE = "preserve"
    DISCARD = "discard"


class FrameRounding(str, Enum):
    """How requested times align to discrete frame boundaries."""

    NEAREST = "nearest"
    FLOOR = "floor"
    CEIL = "ceil"


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TemporalContractError(f"{name} must be an integer")
    if value < minimum:
        raise TemporalContractError(f"{name} must be at least {minimum}")
    return value


def _enum_value(value: object, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise TemporalContractError(f"{name} must be a string")
    try:
        return enum_type(value.strip().casefold())
    except ValueError as exc:
        choices = ", ".join(str(item.value) for item in enum_type)
        raise TemporalContractError(
            f"Unknown {name} {value!r}; choose {choices}"
        ) from exc


def _seconds(value: object, name: str, *, allow_zero: bool) -> float:
    if isinstance(value, bool):
        raise TemporalContractError(f"{name} must be a number")
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise TemporalContractError(f"{name} must be a number") from exc
    lower_bound = 0 if allow_zero else 0.0
    if not math.isfinite(numeric) or numeric < lower_bound:
        qualifier = "non-negative" if allow_zero else "positive"
        raise TemporalContractError(f"{name} must be a {qualifier} finite number")
    if not allow_zero and numeric == 0:
        raise TemporalContractError(f"{name} must be a positive finite number")
    if numeric > MAX_TIMELINE_SECONDS:
        raise TemporalContractError(
            f"{name} cannot exceed {MAX_TIMELINE_SECONDS} seconds"
        )
    return numeric


def _as_fraction(value: int | float) -> Fraction:
    """Convert user-facing decimal values without importing binary-float noise."""

    return Fraction(str(value))


def _round_fraction(value: Fraction, policy: FrameRounding) -> int:
    if policy is FrameRounding.FLOOR:
        return value.numerator // value.denominator
    if policy is FrameRounding.CEIL:
        return -(-value.numerator // value.denominator)
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def _format_fraction(value: Fraction, *, places: int = 9) -> str:
    rendered = f"{float(value):.{places}f}".rstrip("0").rstrip(".")
    return rendered or "0"


@dataclass(frozen=True, slots=True)
class FrameRate:
    """An exact positive frame rate such as NTSC ``30000/1001``."""

    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        numerator = _integer(self.numerator, "frame-rate numerator", minimum=1)
        denominator = _integer(
            self.denominator,
            "frame-rate denominator",
            minimum=1,
        )
        normalized = Fraction(numerator, denominator)
        if normalized > MAX_FRAME_RATE:
            raise TemporalContractError(
                f"frame rate cannot exceed {MAX_FRAME_RATE} frames per second"
            )
        if normalized.denominator > MAX_FRAME_RATE_DENOMINATOR:
            raise TemporalContractError(
                f"frame-rate denominator cannot exceed {MAX_FRAME_RATE_DENOMINATOR}"
            )
        object.__setattr__(self, "numerator", normalized.numerator)
        object.__setattr__(self, "denominator", normalized.denominator)

    @classmethod
    def parse(cls, value: object) -> "FrameRate":
        """Parse another rate, JSON object, ratio, decimal, or number."""

        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            keys = set(value)
            expected = {"numerator", "denominator"}
            if keys != expected:
                raise TemporalContractError(
                    "Serialized frame rate must contain numerator and denominator"
                )
            return cls(value["numerator"], value["denominator"])  # type: ignore[arg-type]
        if isinstance(value, bool):
            raise TemporalContractError("frame rate must be a number or ratio")
        try:
            text = str(value).strip()
            ratio = Fraction(text)
        except (AttributeError, ValueError, ZeroDivisionError) as exc:
            raise TemporalContractError(
                "frame rate must be a number or ratio such as 30000/1001"
            ) from exc
        if "/" not in text:
            ratio = ratio.limit_denominator(MAX_FRAME_RATE_DENOMINATOR)
        return cls(ratio.numerator, ratio.denominator)

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    @property
    def fps(self) -> float:
        return self.numerator / self.denominator

    @property
    def ffmpeg_value(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    def to_dict(self) -> dict[str, int]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FrameRate":
        return cls.parse(values)


@dataclass(frozen=True, slots=True)
class TemporalRenderRequest:
    """Portable timing and audio intent for one temporal render.

    ``frame_rate`` is optional: omitted requests retain the decoded source
    cadence, while an explicit exact rate repairs unreliable metadata or
    deliberately re-times the decoded frame sequence.
    """

    start: float = 0.0
    duration: float | None = None
    frame_rate: FrameRate | str | int | float | Mapping[str, Any] | None = None
    audio: AudioPolicy | str = AudioPolicy.PRESERVE
    rounding: FrameRounding | str = FrameRounding.NEAREST
    contract_version: int = field(
        default=TEMPORAL_CONTRACT_VERSION,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.contract_version, bool)
            or not isinstance(self.contract_version, int)
            or self.contract_version != TEMPORAL_CONTRACT_VERSION
        ):
            raise TemporalContractError(
                "Unsupported temporal contract version "
                f"{self.contract_version}; expected {TEMPORAL_CONTRACT_VERSION}"
            )
        start = _seconds(self.start, "start", allow_zero=True)
        duration = (
            None
            if self.duration is None
            else _seconds(self.duration, "duration", allow_zero=False)
        )
        rate = None if self.frame_rate is None else FrameRate.parse(self.frame_rate)
        audio = _enum_value(self.audio, AudioPolicy, "audio policy")
        rounding = _enum_value(self.rounding, FrameRounding, "frame rounding")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "frame_rate", rate)
        object.__setattr__(self, "audio", audio)
        object.__setattr__(self, "rounding", rounding)

    @property
    def selected_frame_rate(self) -> FrameRate | None:
        return cast(FrameRate | None, self.frame_rate)

    @property
    def audio_policy(self) -> AudioPolicy:
        return cast(AudioPolicy, self.audio)

    @property
    def rounding_policy(self) -> FrameRounding:
        return cast(FrameRounding, self.rounding)

    def resolve(
        self,
        source_frame_rate: FrameRate | str | int | float | Mapping[str, Any],
        source_frames: int | None = None,
    ) -> "ResolvedTimeline":
        """Resolve seconds into deterministic integer frame boundaries."""

        rate = self.selected_frame_rate or FrameRate.parse(source_frame_rate)
        if source_frames is not None:
            _integer(source_frames, "source_frames")
        start_frame = _round_fraction(
            _as_fraction(self.start) * rate.fraction,
            self.rounding_policy,
        )
        available = (
            None if source_frames is None else max(0, source_frames - start_frame)
        )
        if self.duration is None:
            frame_count = available
        else:
            requested = max(
                1,
                _round_fraction(
                    _as_fraction(self.duration) * rate.fraction,
                    self.rounding_policy,
                ),
            )
            frame_count = requested if available is None else min(requested, available)
        return ResolvedTimeline(
            frame_rate=rate,
            start_frame=start_frame,
            frame_count=frame_count,
            source_frames=source_frames,
            requested_start=self.start,
            requested_duration=self.duration,
            audio=self.audio_policy,
            rounding=self.rounding_policy,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "duration": self.duration,
            "frame_rate": (
                self.selected_frame_rate.to_dict()
                if self.selected_frame_rate is not None
                else None
            ),
            "audio": self.audio_policy.value,
            "rounding": self.rounding_policy.value,
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TemporalRenderRequest":
        if not isinstance(values, Mapping):
            raise TemporalContractError("Serialized temporal request must be an object")
        try:
            return cls(**dict(values))
        except TypeError as exc:
            raise TemporalContractError(
                f"Malformed serialized temporal request: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class ResolvedTimeline:
    """Exact frame boundaries shared by decode, encode, audio, and progress."""

    frame_rate: FrameRate
    start_frame: int
    frame_count: int | None
    source_frames: int | None
    requested_start: float
    requested_duration: float | None
    audio: AudioPolicy
    rounding: FrameRounding

    def __post_init__(self) -> None:
        if not isinstance(self.frame_rate, FrameRate):
            raise TemporalContractError("frame_rate must be a FrameRate")
        _integer(self.start_frame, "start_frame")
        if self.frame_count is not None:
            _integer(self.frame_count, "frame_count")
        if self.source_frames is not None:
            _integer(self.source_frames, "source_frames")
        if not isinstance(self.audio, AudioPolicy):
            raise TemporalContractError("audio must be an AudioPolicy")
        if not isinstance(self.rounding, FrameRounding):
            raise TemporalContractError("rounding must be a FrameRounding")

    @property
    def end_frame(self) -> int | None:
        if self.frame_count is None:
            return None
        return self.start_frame + self.frame_count

    @property
    def aligned_start(self) -> Fraction:
        return Fraction(
            self.start_frame * self.frame_rate.denominator,
            self.frame_rate.numerator,
        )

    @property
    def encoded_duration(self) -> Fraction | None:
        if self.frame_count is None:
            return None
        return Fraction(
            self.frame_count * self.frame_rate.denominator,
            self.frame_rate.numerator,
        )

    @property
    def aligned_start_seconds(self) -> float:
        return float(self.aligned_start)

    @property
    def encoded_seconds(self) -> float | None:
        duration = self.encoded_duration
        return None if duration is None else float(duration)

    @property
    def ffmpeg_start(self) -> str:
        return _format_fraction(self.aligned_start)

    @property
    def ffmpeg_duration(self) -> str | None:
        duration = self.encoded_duration
        return None if duration is None else _format_fraction(duration)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_rate": self.frame_rate.to_dict(),
            "start_frame": self.start_frame,
            "frame_count": self.frame_count,
            "end_frame": self.end_frame,
            "source_frames": self.source_frames,
            "requested_start": self.requested_start,
            "requested_duration": self.requested_duration,
            "aligned_start_seconds": self.aligned_start_seconds,
            "encoded_seconds": self.encoded_seconds,
            "audio": self.audio.value,
            "rounding": self.rounding.value,
            "contract_version": TEMPORAL_CONTRACT_VERSION,
        }


__all__ = [
    "AudioPolicy",
    "FrameRate",
    "FrameRounding",
    "MAX_FRAME_RATE",
    "MAX_FRAME_RATE_DENOMINATOR",
    "MAX_TIMELINE_SECONDS",
    "ResolvedTimeline",
    "TEMPORAL_CONTRACT_VERSION",
    "TemporalContractError",
    "TemporalRenderRequest",
]

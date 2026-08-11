"""Tests for the streamed full-colour glyph video pipeline."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from glyph_forge.cli import app
from glyph_forge.live import video
from glyph_forge.live.video import (
    GlyphVideoRenderer,
    VideoExportConfig,
    VideoExportResult,
    build_ffmpeg_command,
    export_glyph_video,
)
from glyph_forge.temporal import AudioPolicy, FrameRate


def test_adaptive_video_profiles_scale_cleanly() -> None:
    eco = VideoExportConfig.adaptive("eco")
    workstation = VideoExportConfig.adaptive("workstation")

    assert eco.width < workstation.width
    assert eco.columns < workstation.columns
    assert eco.workers == 1
    assert workstation.workers > eco.workers
    assert eco.width % eco.columns == 0
    assert workstation.height % workstation.rows == 0


def test_named_language_charsets_are_resolved() -> None:
    assert video._resolve_charset("greek").startswith("αβγ")


def test_video_charset_uses_strict_shared_preset_and_literal_syntax() -> None:
    assert video._resolve_charset("literal:abc") == "abc"
    with pytest.raises(ValueError, match="did you mean detailed"):
        video._resolve_charset("detaled")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"width": 7}, "even"),
        ({"duration": 0}, "Duration"),
        ({"start": -1}, "Start"),
        ({"crf": 52}, "CRF"),
        ({"preset": ""}, "preset"),
        ({"workers": 0}, "Worker"),
        ({"workers": 65}, "Worker"),
    ],
)
def test_video_config_validation(changes: dict[str, Any], message: str) -> None:
    base = {
        "width": 16,
        "height": 16,
        "columns": 2,
        "rows": 2,
        "charset": " @",
    }
    with pytest.raises(ValueError, match=message):
        VideoExportConfig(**(base | changes)).validated()


def test_full_colour_renderer_maps_brightness_and_preserves_colour() -> None:
    renderer = GlyphVideoRenderer(
        VideoExportConfig(
            width=16,
            height=16,
            columns=2,
            rows=2,
            charset="@",
        )
    )
    sampled = np.zeros((2, 2, 3), dtype=np.uint8)
    sampled[:, 1] = [255, 0, 0]

    rendered = renderer.render_sampled_rgb(sampled)

    assert rendered.shape == (16, 16, 3)
    assert rendered.dtype == np.uint8
    assert rendered[:, :8].sum() == 0
    assert rendered[:, 8:, 0].sum() > 0
    assert rendered[:, 8:, 1:].sum() == 0


def test_video_output_pixels_are_independent_from_glyph_grid() -> None:
    config = VideoExportConfig(
        width=10,
        height=8,
        columns=3,
        rows=3,
        charset="@",
        brightness=1.0,
        contrast=1.0,
    ).validated()
    renderer = GlyphVideoRenderer(config)

    rendered = renderer.render_sampled_rgb(np.full((3, 3, 3), 128, dtype=np.uint8))

    assert not config.uses_integral_cells
    assert rendered.shape == (8, 10, 3)


def test_video_tone_is_normalized_once_during_validation() -> None:
    selected = VideoExportConfig(
        width=16,
        height=16,
        columns=2,
        rows=2,
        charset="@",
        brightness=3.0,
        contrast=-1.0,
    ).validated()

    assert selected.brightness == 2.0
    assert selected.contrast == 0.0


def test_ffmpeg_command_retains_script_audio_and_quality_options() -> None:
    config = VideoExportConfig(
        width=1920,
        height=1080,
        columns=160,
        rows=90,
        start=2.5,
        duration=4.25,
        crf=12,
        preset="slow",
    )

    rate = FrameRate.parse(29.97)
    timeline = config.temporal_request().resolve(rate)
    command = build_ffmpeg_command(
        "source.mov",
        "output.mp4",
        config,
        rate,
        timeline=timeline,
    )

    assert command[:2] == ["ffmpeg", "-hide_banner"]
    assert ["-video_size", "1920x1080"] == command[
        command.index("-video_size") : command.index("-video_size") + 2
    ]
    assert command[command.index("-framerate") + 1] == "2997/100"
    assert command[command.index("-ss") + 1] == timeline.ffmpeg_start
    assert command[command.index("-t") + 1] == timeline.ffmpeg_duration
    assert command[command.index("-preset") + 1] == "slow"
    assert command[command.index("-crf") + 1] == "12"
    assert "1:a:0?" in command
    assert "-shortest" in command
    assert "+faststart" in command
    assert command[-1] == "output.mp4"


def test_ffmpeg_command_can_create_a_silent_video_without_source_muxing() -> None:
    config = VideoExportConfig(
        width=16,
        height=16,
        columns=2,
        rows=2,
        audio="discard",
    )

    command = build_ffmpeg_command("source.mov", "output.mp4", config, 30)

    assert "source.mov" not in command
    assert "-an" in command
    assert "-ss" not in command
    assert "1:a:0?" not in command


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = iter(frames)
        self.released = False
        self.seek = 0

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return {1: 10.0, 2: 2.0}.get(property_id, 0.0)

    def set(self, property_id: int, value: int) -> bool:
        self.seek = value
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self) -> None:
        self.released = True


class _FakeCv2:
    CAP_PROP_FPS = 1
    CAP_PROP_FRAME_COUNT = 2
    CAP_PROP_POS_FRAMES = 3
    INTER_AREA = 4

    def __init__(self, capture: _FakeCapture) -> None:
        self.capture = capture

    def VideoCapture(self, _path: str) -> _FakeCapture:
        return self.capture

    @staticmethod
    def resize(
        frame: np.ndarray, size: tuple[int, int], interpolation: int
    ) -> np.ndarray:
        del interpolation
        return np.resize(frame, (size[1], size[0], 3)).astype(np.uint8)


def test_parallel_frame_rendering_is_bounded_and_ordered() -> None:
    frames = [np.full((1, 1, 3), marker, dtype=np.uint8) for marker in (1, 2, 3, 4)]
    capture = _FakeCapture(frames)
    barrier = threading.Barrier(2)

    class Renderer:
        def __init__(self) -> None:
            self.active = 0
            self.maximum_active = 0
            self.lock = threading.Lock()

        def render_bgr(self, frame: np.ndarray, _backend: Any) -> np.ndarray:
            marker = int(frame[0, 0, 0])
            with self.lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            try:
                if marker <= 2:
                    barrier.wait(timeout=2)
                return np.full((1, 1, 3), marker, dtype=np.uint8)
            finally:
                with self.lock:
                    self.active -= 1

    renderer = Renderer()
    rendered = list(
        video._iter_rendered_frames(
            capture,
            renderer,  # type: ignore[arg-type]
            _FakeCv2(capture),
            total_frames=None,
            workers=2,
        )
    )

    assert [int(frame[0, 0, 0]) for frame in rendered] == [1, 2, 3, 4]
    assert renderer.maximum_active == 2


class _FakePipe:
    def __init__(self) -> None:
        self.bytes_written = 0
        self.closed = False

    def write(self, data: bytes) -> int:
        self.bytes_written += len(data)
        return len(data)

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    instances: list["_FakeProcess"] = []

    def __init__(
        self,
        command: list[str],
        stdin: Any = None,
        env: dict[str, str] | None = None,
    ) -> None:
        del stdin
        self.command = command
        self.environment = env
        self.stdin = _FakePipe()
        self.returncode: int | None = None
        Path(command[-1]).write_bytes(b"encoded-video")
        self.instances.append(self)

    def wait(self, timeout: int | None = None) -> int:
        del timeout
        self.returncode = 0
        return 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def test_export_streams_frames_atomically_without_an_image_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mov"
    destination = tmp_path / "result.mp4"
    source.write_bytes(b"source")
    frames = [
        np.full((2, 2, 3), [0, 0, 255], dtype=np.uint8),
        np.full((2, 2, 3), [255, 0, 0], dtype=np.uint8),
    ]
    capture = _FakeCapture(frames)
    monkeypatch.setattr(video, "_load_opencv", lambda: _FakeCv2(capture))
    monkeypatch.setattr(video, "_executable_available", lambda _name: True)
    monkeypatch.setattr(video.subprocess, "Popen", _FakeProcess)
    monkeypatch.setattr(video, "subprocess_environment", lambda: {"SAFE": "1"})
    progress = []

    result = export_glyph_video(
        source,
        destination,
        VideoExportConfig(
            width=16,
            height=16,
            columns=2,
            rows=2,
            charset=" @",
        ),
        progress=progress.append,
    )

    process = _FakeProcess.instances[-1]
    assert result.rendered_frames == 2
    assert result.fps == 10.0
    assert result.rendered_seconds == 0.2
    assert result.workers == 1
    assert result.source == source
    assert result.source_bytes == len(b"source")
    assert result.output_bytes == len(b"encoded-video")
    assert result.timeline is not None
    assert result.timeline.frame_rate == FrameRate(10)
    assert result.timeline.frame_count == 2
    assert result.timeline.audio is AudioPolicy.PRESERVE
    assert result.raw_rgb_bytes == 2 * 16 * 16 * 3
    assert result.to_dict()["render_fps"] > 0
    assert destination.read_bytes() == b"encoded-video"
    assert process.stdin.bytes_written == 2 * 16 * 16 * 3
    assert process.stdin.closed
    assert process.environment == {"SAFE": "1"}
    assert capture.released
    assert [item.rendered_frames for item in progress] == [1, 2]
    assert not list(tmp_path.glob(".*.glyph-forge-*.partial*"))


def test_export_refuses_to_overwrite_its_input(tmp_path: Path) -> None:
    source = tmp_path / "only-copy.mp4"
    source.write_bytes(b"video")

    with pytest.raises(video.VideoExportError, match="must be different"):
        export_glyph_video(
            source,
            source,
            VideoExportConfig(
                width=16,
                height=16,
                columns=2,
                rows=2,
                charset="@",
            ),
        )


def test_cli_video_preserves_every_standalone_script_option(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mov"
    destination = tmp_path / "custom.mp4"
    font = tmp_path / "font.ttf"
    source.write_bytes(b"source")
    font.write_bytes(b"font")
    calls: list[tuple[Path, Path, VideoExportConfig]] = []

    def fake_export(
        input_path: Path,
        output_path: Path,
        config: VideoExportConfig,
        **_kwargs: Any,
    ) -> VideoExportResult:
        calls.append((Path(input_path), Path(output_path), config))
        return VideoExportResult(
            output=Path(output_path),
            rendered_frames=24,
            fps=24.0,
            elapsed=1.0,
            width=config.width,
            height=config.height,
            columns=config.columns,
            rows=config.rows,
        )

    monkeypatch.setattr(video, "export_glyph_video", fake_export)
    result = CliRunner().invoke(
        app,
        [
            "video",
            str(source),
            str(destination),
            "--width",
            "640",
            "--height",
            "360",
            "--columns",
            "80",
            "--rows",
            "45",
            "--charset",
            "blocks",
            "--font",
            str(font),
            "--start",
            "1.5",
            "--duration",
            "2",
            "--frame-rate",
            "30000/1001",
            "--no-audio",
            "--frame-rounding",
            "floor",
            "--crf",
            "20",
            "--preset",
            "fast",
            "--ffmpeg",
            "custom-ffmpeg",
            "--workers",
            "3",
            "--brightness",
            "1.25",
            "--contrast",
            "1.15",
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    _, called_output, config = calls[0]
    assert called_output == destination.resolve()
    assert (config.width, config.height, config.columns, config.rows) == (
        640,
        360,
        80,
        45,
    )
    assert config.charset == "blocks"
    assert config.font == str(font)
    assert config.start == 1.5
    assert config.duration == 2
    assert config.frame_rate == FrameRate(30_000, 1_001)
    assert config.audio is AudioPolicy.DISCARD
    assert config.rounding.value == "floor"
    assert config.crf == 20
    assert config.preset == "fast"
    assert config.ffmpeg == "custom-ffmpeg"
    assert config.workers == 3
    assert config.brightness == 1.25
    assert config.contrast == 1.15


def test_cli_video_chooses_a_shareable_default_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "party.mov"
    source.write_bytes(b"source")
    outputs: list[Path] = []

    def fake_export(
        _input: Path,
        output: Path,
        config: VideoExportConfig,
        **_kwargs: Any,
    ) -> VideoExportResult:
        outputs.append(Path(output))
        return VideoExportResult(
            Path(output),
            1,
            30.0,
            0.1,
            config.width,
            config.height,
            config.columns,
            config.rows,
        )

    monkeypatch.setattr(video, "export_glyph_video", fake_export)
    result = CliRunner().invoke(app, ["video", str(source), "--quiet"])

    assert result.exit_code == 0, result.output
    assert outputs == [tmp_path / "party.glyph.mp4"]


def test_cli_video_emits_machine_readable_performance_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")

    def fake_export(
        input_path: Path,
        output_path: Path,
        config: VideoExportConfig,
        **_kwargs: Any,
    ) -> VideoExportResult:
        return VideoExportResult(
            output=Path(output_path),
            rendered_frames=60,
            fps=30.0,
            elapsed=1.0,
            width=config.width,
            height=config.height,
            columns=config.columns,
            rows=config.rows,
            workers=config.workers,
            source=Path(input_path),
            source_bytes=100,
            output_bytes=250,
        )

    monkeypatch.setattr(video, "export_glyph_video", fake_export)
    result = CliRunner().invoke(
        app,
        ["video", str(source), "--performance", "eco", "--json"],
    )

    assert result.exit_code == 0, result.output
    metrics = json.loads(result.stdout)
    assert metrics["render_fps"] == 60.0
    assert metrics["rendered_seconds"] == 2.0
    assert metrics["realtime_factor"] == 0.5
    assert metrics["output_source_ratio"] == 2.5
    assert metrics["workers"] == 1

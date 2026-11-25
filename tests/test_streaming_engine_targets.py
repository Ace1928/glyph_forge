"""Targeted tests for unified streaming engine paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Unified engine setup path constructs streaming renderer tables requiring cv2.
pytest.importorskip("cv2", reason="streaming engine tests require opencv-python")


def test_engine_setup_pipeline(monkeypatch, tmp_path):
    from glyph_forge.streaming import engine as eng
    from glyph_forge.streaming.extractors import ExtractionResult

    class DummyExtractor:
        def extract(self, *args, **kwargs):
            return ExtractionResult(
                video_url=str(tmp_path / "video.mp4"),
                audio_url=None,
                title="Demo",
                duration=1.0,
                fps=30.0,
                format="local_file",
            )

    monkeypatch.setattr(eng, "VideoSourceExtractor", DummyExtractor)

    config = eng.UnifiedStreamConfig(
        source=str(tmp_path / "video.mp4"),
        audio_enabled=False,
        record_enabled=True,
        record_path=str(tmp_path / "out.mp4"),
    )
    engine = eng.UnifiedStreamEngine(config)
    monkeypatch.setattr(engine, "_start_buffering_thread", lambda: None)

    engine._setup_pipeline()
    assert engine.source_path == str(tmp_path / "video.mp4")
    assert engine.recorder is not None
    engine._cleanup()


def test_engine_cleanup_writes_metadata(tmp_path):
    from glyph_forge.streaming import engine as eng

    config = eng.UnifiedStreamConfig(source="video.mp4", audio_enabled=False)
    engine = eng.UnifiedStreamEngine(config)
    output = tmp_path / "out.mp4"
    output.write_bytes(b"")
    engine.last_recording_path = output
    engine.extraction_info = SimpleNamespace(
        title="Demo",
        video_url="video.mp4",
        audio_url=None,
        duration=1.0,
        fps=30.0,
        width=None,
        height=None,
        format="local_file",
        is_live=False,
    )
    engine.buffer = SimpleNamespace(stop=lambda: None)
    engine.recorder = None
    engine.audio = None

    engine._cleanup()
    assert output.with_suffix(".metadata.json").exists()


def test_engine_selects_ffmpeg_backend(monkeypatch):
    from glyph_forge.streaming import engine as eng

    config = eng.UnifiedStreamConfig(source="video.mp4", frame_backend="auto")
    engine = eng.UnifiedStreamEngine(config)
    engine.extraction_info = SimpleNamespace(format="local_file", is_live=False)
    monkeypatch.setattr(engine, "_ffmpeg_available", lambda: True)

    assert engine._select_frame_backend() == "ffmpeg"


def test_engine_uses_cv2_for_webcam(monkeypatch):
    from glyph_forge.streaming import engine as eng

    config = eng.UnifiedStreamConfig(source=0, frame_backend="auto")
    engine = eng.UnifiedStreamEngine(config)
    engine.extraction_info = SimpleNamespace(format="webcam", is_live=True)
    monkeypatch.setattr(engine, "_ffmpeg_available", lambda: True)

    assert engine._select_frame_backend() == "cv2"


def test_engine_cleanup_muxes_audio_when_recorded(tmp_path):
    from glyph_forge.streaming import engine as eng

    class DummyRecorder:
        def __init__(self, output_path):
            self.config = SimpleNamespace(output_path=output_path)
            self.muxed = False

        def close(self):
            return None

        def mux_audio(self, _audio_path):
            self.muxed = True
            return self.config.output_path

    video_path = tmp_path / "out.mp4"
    video_path.write_bytes(b"")
    audio_path = tmp_path / "audio.m4a"
    audio_path.write_bytes(b"audio")

    config = eng.UnifiedStreamConfig(
        source="https://www.youtube.com/watch?v=demo",
        audio_enabled=True,
        record_enabled=True,
        mux_audio=True,
    )
    engine = eng.UnifiedStreamEngine(config)
    engine.buffer = SimpleNamespace(stop=lambda: None)
    engine.recorder = DummyRecorder(video_path)
    engine.audio = None
    engine.audio_source_for_mux = "https://www.youtube.com/watch?v=demo"
    engine.audio_path = audio_path
    engine.last_recording_path = video_path
    engine.extraction_info = SimpleNamespace(
        title="Demo",
        video_url="https://www.youtube.com/watch?v=demo",
        audio_url="https://example.com/audio",
        duration=1.0,
        fps=30.0,
        width=None,
        height=None,
        format="youtube",
        is_live=False,
    )

    engine._cleanup()
    assert engine.recorder.muxed is True


def test_engine_mux_prefers_local_audio(tmp_path):
    from glyph_forge.streaming import engine as eng

    audio_path = tmp_path / "audio.m4a"
    audio_path.write_bytes(b"audio")

    config = eng.UnifiedStreamConfig(source="https://www.youtube.com/watch?v=demo")
    engine = eng.UnifiedStreamEngine(config)
    info = SimpleNamespace(audio_url=str(audio_path))

    assert engine._select_audio_source_for_mux(info) == str(audio_path)

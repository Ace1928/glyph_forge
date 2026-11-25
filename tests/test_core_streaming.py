"""
Tests for Glyph Forge streaming core modules.

Comprehensive tests for:
- StreamConfig
- AdaptiveBuffer
- VideoCapture
- GlyphRenderer
- GlyphRecorder
- AudioSync
"""

import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

# Streaming renderer path requires OpenCV; skip module when unavailable.
cv2 = pytest.importorskip("cv2", reason="glyph_forge streaming tests require opencv-python")

# Import modules under test
from glyph_forge.streaming.core.buffer import (
    AdaptiveBuffer,
    BufferedFrame,
    BufferMetrics,
)
from glyph_forge.streaming.core.config import (
    ColorMode,
    RenderMode,
    StreamConfig,
)
from glyph_forge.streaming.core.recorder import GlyphRecorder, RecorderConfig
from glyph_forge.streaming.core.renderer import (
    EXTENDED_GRADIENT,
    GlyphRenderer,
    LookupTables,
    RenderConfig,
)
from glyph_forge.streaming.core.sync import AudioSync

# ═══════════════════════════════════════════════════════════════
# StreamConfig Tests
# ═══════════════════════════════════════════════════════════════


class TestStreamConfig:
    """Tests for StreamConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StreamConfig()

        assert config.resolution == "auto"
        assert config.max_resolution == 720
        assert config.render_mode == RenderMode.GRADIENT
        assert config.color_mode == ColorMode.ANSI256
        assert config.record_output
        assert config.audio_enabled

    def test_terminal_size(self):
        """Test terminal size detection."""
        config = StreamConfig()
        width, height = config.get_terminal_size()

        assert width > 0
        assert height > 0

    def test_display_size_calculation(self):
        """Test display size calculation preserves aspect ratio."""
        config = StreamConfig()

        # 16:9 video
        w, h = config.calculate_display_size(1920, 1080)

        assert w > 0
        assert h > 0
        # Should be constrained to terminal size

    def test_record_size_calculation(self):
        """Test recording size calculation."""
        config = StreamConfig()

        # 1080p source
        w, h = config.calculate_record_size(1920, 1080)
        assert h == 90  # HD recording height

        # 720p source
        w, h = config.calculate_record_size(1280, 720)
        assert h == 60

    def test_output_path_generation(self):
        """Test output path generation."""
        config = StreamConfig()

        # YouTube URL
        path = config.generate_output_path("https://youtube.com/watch?v=dQw4w9WgXcQ")
        assert "youtube_dQw4w9WgXcQ" in str(path)
        assert path.suffix == ".mp4"

        # Local file
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
            path = config.generate_output_path(tf.name)
            assert Path(tf.name).stem in str(path)


# ═══════════════════════════════════════════════════════════════
# AdaptiveBuffer Tests
# ═══════════════════════════════════════════════════════════════


class TestAdaptiveBuffer:
    """Tests for AdaptiveBuffer."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = AdaptiveBuffer(target_fps=30.0, min_buffer_seconds=5.0, target_buffer_seconds=30.0)

        assert buffer.target_fps == 30.0
        assert buffer.buffer_size == 0
        assert buffer.buffer_seconds == 0.0

    def test_buffer_metrics(self):
        """Test buffer metrics tracking."""
        metrics = BufferMetrics()

        # Add samples
        metrics.add_sample(0.01)  # 100 fps
        metrics.add_sample(0.02)  # 50 fps

        assert len(metrics.render_times) == 2
        assert 0.01 < metrics.avg_render_time < 0.02
        assert metrics.max_render_time == 0.02
        assert metrics.render_fps > 50

    def test_min_required_frames(self):
        """Test minimum required frames calculation."""
        buffer = AdaptiveBuffer(target_fps=30.0, min_buffer_seconds=5.0, target_buffer_seconds=30.0)
        buffer._total_frames = 1000

        # Fast renderer
        buffer.metrics.add_sample(0.001)  # 1000 fps
        min_frames = buffer.min_required_frames
        assert min_frames == 150  # 5 seconds * 30 fps

        # Slow renderer
        buffer.metrics.render_times.clear()
        buffer.metrics.add_sample(0.1)  # 10 fps (slower than playback)
        min_frames = buffer.min_required_frames
        assert min_frames > 150  # Need more buffer

    def test_ready_for_playback(self):
        """Test playback readiness detection."""
        buffer = AdaptiveBuffer(target_fps=30.0, min_buffer_seconds=1.0)
        buffer._total_frames = 100
        buffer.metrics.add_sample(0.001)  # Fast render

        # Not ready initially
        assert not buffer.ready_for_playback

        # Add frames
        for i in range(30):  # 1 second worth
            frame = BufferedFrame(
                display_data=f"frame_{i}",
                record_data=None,
                timestamp=i / 30.0,
                frame_idx=i,
            )
            buffer._frames.append(frame)

        # Should be ready now
        assert buffer.ready_for_playback

    def test_get_next_frame(self):
        """Test frame retrieval."""
        buffer = AdaptiveBuffer(target_fps=30.0)
        buffer._buffering_complete = True

        # Add a frame
        frame = BufferedFrame(
            display_data="test_frame",
            record_data="test_record",
            timestamp=0.0,
            frame_idx=0,
        )
        buffer._frames.append(frame)

        # Retrieve it
        retrieved = buffer.get_next_frame(timeout=0.1)
        assert retrieved is not None
        assert retrieved.display_data == "test_frame"

        # Buffer should be empty now
        assert buffer.buffer_size == 0

        # Next retrieval should return None
        retrieved = buffer.get_next_frame(timeout=0.1)
        assert retrieved is None


# ═══════════════════════════════════════════════════════════════
# GlyphRenderer Tests
# ═══════════════════════════════════════════════════════════════


class TestGlyphRenderer:
    """Tests for GlyphRenderer."""

    @pytest.fixture
    def renderer(self, tmp_path):
        """Create renderer with temp cache."""
        config = RenderConfig(mode="gradient", color="ansi256", cache_dir=tmp_path)
        return GlyphRenderer(config)

    def test_initialization(self, renderer):
        """Test renderer initialization."""
        assert renderer.tables is not None
        assert len(renderer._char_array) == len(EXTENDED_GRADIENT)

    def test_render_no_color(self, renderer):
        """Test rendering without color."""
        frame = np.random.randint(0, 255, (10, 20, 3), dtype=np.uint8)
        result = renderer.render(frame, width=20, height=10, color="none")

        assert isinstance(result, str)
        lines = result.split("\n")
        assert len(lines) == 10
        assert all(len(line) == 20 for line in lines)
        # No ANSI codes
        assert "\033[" not in result

    def test_render_ansi256(self, renderer):
        """Test rendering with ANSI256 colors."""
        frame = np.random.randint(0, 255, (10, 20, 3), dtype=np.uint8)
        result = renderer.render(frame, width=20, height=10, color="ansi256")

        assert isinstance(result, str)
        # Should contain ANSI escape codes
        assert "\033[38;5;" in result
        # Should end with reset
        assert result.endswith("\033[0m")

    def test_render_truecolor(self, renderer):
        """Test rendering with TrueColor."""
        frame = np.random.randint(0, 255, (10, 20, 3), dtype=np.uint8)
        result = renderer.render(frame, width=20, height=10, color="truecolor")

        assert isinstance(result, str)
        # Should contain TrueColor escape codes
        assert "\033[38;2;" in result

    def test_ascii_mode_contains_only_ascii(self, tmp_path):
        renderer = GlyphRenderer(RenderConfig(mode="ascii", color="none", cache_dir=tmp_path))
        frame = np.random.default_rng(7).integers(0, 256, (8, 16, 3), dtype=np.uint8)

        result = renderer.render(frame, width=16, height=8)

        assert result.isascii()

    @pytest.mark.parametrize(
        ("frame", "error"),
        [
            (np.array([], dtype=np.uint8), "frame must"),
            (np.zeros((2, 2, 2), dtype=np.uint8), "frame must"),
        ],
    )
    def test_invalid_frame_fails_clearly(self, tmp_path, frame, error):
        renderer = GlyphRenderer(RenderConfig(cache_dir=tmp_path))

        with pytest.raises(ValueError, match=error):
            renderer.render(frame)

    def test_render_braille(self, renderer):
        """Test Braille rendering."""
        frame = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)
        result = renderer.render_braille(frame, width=20, height=10)

        assert isinstance(result, str)
        lines = result.split("\n")
        assert len(lines) == 10
        # Check for Braille characters (U+2800 range)
        assert any(ord(c) >= 0x2800 and ord(c) <= 0x28FF for line in lines for c in line)

    def test_gradient_mapping(self, renderer):
        """Test luminance to character mapping."""
        # Dark pixel
        dark = np.zeros((1, 1, 3), dtype=np.uint8)
        result = renderer.render(dark, width=1, height=1, color="none")
        dense_chars = set(renderer.tables.get_gradient_chars()[:10])
        assert result.strip() in dense_chars  # Should be dense char

        # Bright pixel
        bright = np.full((1, 1, 3), 255, dtype=np.uint8)
        result = renderer.render(bright, width=1, height=1, color="none")
        assert result.strip() in EXTENDED_GRADIENT[-10:]  # Should be sparse char


# ═══════════════════════════════════════════════════════════════
# LookupTables Tests
# ═══════════════════════════════════════════════════════════════


class TestLookupTables:
    """Tests for LookupTables."""

    def test_initialization(self, tmp_path):
        """Test lookup table initialization."""
        tables = LookupTables(tmp_path)

        assert tables._luminance_to_char is not None
        assert tables._rgb_to_ansi256 is not None
        assert len(tables._luminance_to_char) == 256

    def test_cache_persistence(self, tmp_path):
        """Test that tables are cached to disk."""
        # Create tables (should save to cache)
        tables1 = LookupTables(tmp_path)
        cache_file = tmp_path / "lookup_tables_v5.npz"
        assert cache_file.exists()

        # Create again (should load from cache)
        tables2 = LookupTables(tmp_path)
        np.testing.assert_array_equal(tables1._luminance_to_char, tables2._luminance_to_char)

    def test_corrupt_cache_is_rebuilt_without_deserialization(self, tmp_path):
        """A corrupt user cache is ignored and safely replaced."""

        cache_file = tmp_path / "lookup_tables_v5.npz"
        cache_file.write_bytes(b"not a NumPy archive")

        tables = LookupTables(tmp_path)

        assert tables._luminance_to_char is not None
        with np.load(cache_file, allow_pickle=False) as cache:
            assert "luminance_to_char" in cache.files

    def test_oversized_cache_archive_is_rebuilt(self, tmp_path):
        """Compressed cache data cannot expand without a strict bound."""

        cache_file = tmp_path / "lookup_tables_v5.npz"
        with zipfile.ZipFile(cache_file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "luminance_to_char.npy",
                b"0" * (LookupTables._MAX_CACHE_BYTES + 1),
            )
            for name in (
                "rgb_to_ansi256.npy",
                "gradient_chars.npy",
                "ansi256_palette_lab.npy",
                "srgb_to_linear.npy",
            ):
                archive.writestr(name, b"0")

        tables = LookupTables(tmp_path)

        assert tables._luminance_to_char is not None
        assert cache_file.stat().st_size < LookupTables._MAX_CACHE_BYTES

    def test_luminance_mapping(self, tmp_path):
        """Test luminance to character index mapping."""
        tables = LookupTables(tmp_path)

        # Test array input
        lum = np.array([0, 127, 255], dtype=np.uint8)
        indices = tables.get_char_index(lum)

        assert len(indices) == 3
        assert indices[0] <= indices[1] <= indices[2]  # Should increase with brightness

    def test_rgb_to_ansi256(self, tmp_path):
        """Test RGB to ANSI256 mapping."""
        tables = LookupTables(tmp_path)

        # Test with simple colors
        r = np.array([[255, 0, 128]], dtype=np.uint8)
        g = np.array([[0, 255, 128]], dtype=np.uint8)
        b = np.array([[0, 0, 128]], dtype=np.uint8)

        codes = tables.get_ansi256(r, g, b)

        assert codes.shape == (1, 3)
        assert all(0 <= c <= 255 for c in codes.flat)


# ═══════════════════════════════════════════════════════════════
# GlyphRecorder Tests
# ═══════════════════════════════════════════════════════════════


class TestGlyphRecorder:
    """Tests for GlyphRecorder."""

    def test_initialization(self, tmp_path):
        """Test recorder initialization."""
        config = RecorderConfig(output_path=tmp_path / "test.mp4", fps=30.0)
        recorder = GlyphRecorder(config)

        assert recorder.config.fps == 30.0
        assert recorder.frame_count == 0
        assert not recorder.is_open

    def test_ansi_parsing(self, tmp_path):
        """Test ANSI escape code parsing."""
        config = RecorderConfig(output_path=tmp_path / "test.mp4")
        recorder = GlyphRecorder(config)

        # Test ANSI256 parsing
        color = recorder._parse_color("38;5;196", (255, 255, 255))
        assert color == recorder._ansi256_palette[196]

        # Test TrueColor parsing
        color = recorder._parse_color("38;2;100;150;200", (255, 255, 255))
        assert color == (100, 150, 200)

        # Test reset
        color = recorder._parse_color("0", (100, 100, 100))
        assert color == recorder.config.default_fg

    def test_strip_ansi(self, tmp_path):
        """Test ANSI code stripping."""
        config = RecorderConfig(output_path=tmp_path / "test.mp4")
        recorder = GlyphRecorder(config)

        text = "\033[38;5;196mHello\033[0m World"
        stripped = recorder._strip_ansi(text)

        assert stripped == "Hello World"
        assert "\033[" not in stripped

    def test_write_frame(self, monkeypatch, tmp_path):
        """Test frame writing without depending on a platform codec."""

        class FakeWriter:
            def __init__(self, output_path, *_args):
                self.output_path = Path(output_path)
                self.frames = []
                self.open = True

            def isOpened(self):
                return self.open

            def write(self, frame):
                self.frames.append(frame)

            def release(self):
                self.output_path.write_bytes(b"fake video")
                self.open = False

        monkeypatch.setattr(cv2, "VideoWriter_fourcc", lambda *_args: 0)
        monkeypatch.setattr(cv2, "VideoWriter", FakeWriter)
        config = RecorderConfig(output_path=tmp_path / "test.mp4", fps=30.0)
        recorder = GlyphRecorder(config)

        # Write some frames
        glyph_string = "\033[38;5;196m████\033[0m\n████"
        recorder.write_frame(glyph_string)
        recorder.write_frame(glyph_string)

        assert recorder.frame_count == 2
        assert recorder.is_open

        recorder.close()
        assert not recorder.is_open
        assert config.output_path.exists()


# ═══════════════════════════════════════════════════════════════
# AudioSync Tests
# ═══════════════════════════════════════════════════════════════


class TestAudioSync:
    """Tests for AudioSync."""

    def test_initialization(self):
        """Test audio player initialization."""
        audio = AudioSync()

        assert not audio.is_playing
        assert audio.get_position() == 0.0

    def test_sync_offset(self):
        """Test sync offset calculation."""
        audio = AudioSync()

        # Simulate playback started 5 seconds ago
        audio._playing = True
        audio._start_time = time.perf_counter() - 5.0
        audio._start_position = 0.0

        # Get position
        pos = audio.get_position()
        assert 4.9 < pos < 5.1  # Allow for timing variance

        # Calculate sync offset
        video_pos = 4.5
        offset = audio.get_sync_offset(video_pos)
        assert offset > 0  # Audio is ahead

    def test_context_manager(self):
        """Test context manager protocol."""
        with AudioSync() as audio:
            assert audio is not None

        # Should have called stop
        assert not audio.is_playing


# ═══════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_render_pipeline(self, tmp_path):
        """Test complete render pipeline."""
        # Create renderer
        config = RenderConfig(mode="gradient", color="ansi256", cache_dir=tmp_path)
        renderer = GlyphRenderer(config)

        # Create test frame (720p)
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        # Render
        start = time.perf_counter()
        result = renderer.render(frame, width=120, height=40)
        elapsed = time.perf_counter() - start

        # Should be fast
        assert elapsed < 0.5  # Less than 500ms

        # Should produce valid output
        lines = result.split("\n")
        assert len(lines) == 40

    def test_buffer_to_renderer_pipeline(self, tmp_path):
        """Test buffer feeding renderer."""
        # Create renderer
        config = RenderConfig(mode="gradient", color="none", cache_dir=tmp_path)
        renderer = GlyphRenderer(config)

        # Create buffer
        buffer = AdaptiveBuffer(target_fps=30.0, min_buffer_seconds=0.5)

        try:
            # Simulate buffering
            def frame_gen():
                for _ in range(10):
                    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                    yield True, frame
                yield False, None

            gen = frame_gen()

            def get_frame():
                return next(gen, (False, None))

            def render(frame, w, h):
                return renderer.render(frame, w, h)

            buffer.start_buffering(
                frame_generator=get_frame,
                render_func=render,
                display_size=(40, 20),
                record_size=None,
                total_frames=10,
            )

            # Wait for buffering
            time.sleep(0.5)

            # Should have buffered frames
            assert buffer.buffer_size > 0
        finally:
            buffer.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

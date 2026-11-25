"""
Glyph Recording Module for Glyph Forge.

Records terminal glyph output to video files:
- Parses ANSI escape codes to extract colors
- Renders glyphs to images using PIL/Pillow
- Encodes to video with OpenCV
- Muxes audio with ffmpeg
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:  # Optional dependency (streaming extra)
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class RecorderConfig:
    """Recorder configuration."""

    output_path: Path
    fps: float = 30.0
    font_size: int = 14
    font_name: str = "DejaVuSansMono"
    bg_color: Tuple[int, int, int] = (0, 0, 0)  # Black background
    default_fg: Tuple[int, int, int] = (255, 255, 255)  # White foreground
    char_width: int = 8  # Character width in pixels
    char_height: int = 16  # Character height in pixels


class GlyphRecorder:
    """Records terminal glyph output to video.

    Captures ANSI-colored glyph strings and renders them to video frames,
    creating a high-quality recording of the terminal output.

    Usage:
        recorder = GlyphRecorder(RecorderConfig(
            output_path=Path('output.mp4'),
            fps=30.0
        ))

        for glyph_string in render_frames():
            recorder.write_frame(glyph_string)

        recorder.close()
        recorder.mux_audio('audio.m4a')  # Add audio
    """

    # ANSI escape code patterns
    ANSI_PATTERN = re.compile(r"\033\[([0-9;]*)m")
    ANSI_256_FG = re.compile(r"38;5;(\d+)")
    ANSI_TRUE_FG = re.compile(r"38;2;(\d+);(\d+);(\d+)")

    def __init__(self, config: RecorderConfig):
        """Initialize recorder.

        Args:
            config: Recorder configuration
        """
        self.config = config
        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_count = 0
        self._frame_size: Optional[Tuple[int, int]] = None

        # Font setup
        self._font = None
        self._font_metrics = None
        self._setup_font()

        # Build ANSI256 color palette
        self._ansi256_palette = self._build_ansi256_palette()

    def _setup_font(self):
        """Set up font for rendering."""
        try:
            from PIL import ImageFont

            # Try to load monospace font
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
                "/System/Library/Fonts/Menlo.ttc",
                "C:\\Windows\\Fonts\\consola.ttf",
            ]

            for path in font_paths:
                if Path(path).exists():
                    self._font = ImageFont.truetype(path, self.config.font_size)
                    break

            if self._font is None:
                self._font = ImageFont.load_default()

        except Exception as e:
            print(f"Font setup warning: {e}")
            self._font = None

    def _build_ansi256_palette(self) -> List[Tuple[int, int, int]]:
        """Build ANSI256 color palette."""
        palette = []

        # Standard 16 colors (0-15)
        standard = [
            (0, 0, 0),
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            (0, 128, 128),
            (192, 192, 192),
            (128, 128, 128),
            (255, 0, 0),
            (0, 255, 0),
            (255, 255, 0),
            (0, 0, 255),
            (255, 0, 255),
            (0, 255, 255),
            (255, 255, 255),
        ]
        palette.extend(standard)

        # 6x6x6 color cube (16-231)
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    R = 0 if r == 0 else 55 + r * 40
                    G = 0 if g == 0 else 55 + g * 40
                    B = 0 if b == 0 else 55 + b * 40
                    palette.append((R, G, B))

        # Grayscale (232-255)
        for i in range(24):
            gray = 8 + i * 10
            palette.append((gray, gray, gray))

        return palette

    def _init_writer(self, width: int, height: int):
        """Initialize video writer with frame size."""
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for video recording.")
        # Calculate pixel dimensions
        pixel_w = width * self.config.char_width
        pixel_h = height * self.config.char_height

        self._frame_size = (pixel_w, pixel_h)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.config.output_path), fourcc, self.config.fps, self._frame_size
        )

        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {self.config.output_path}")

    def write_frame(self, glyph_string: str):
        """Write a glyph frame to video.

        Args:
            glyph_string: ANSI-colored glyph string
        """
        # Parse dimensions from string
        lines = glyph_string.split("\n")
        height = len(lines)
        width = max(len(self._strip_ansi(line)) for line in lines) if lines else 80

        # Initialize writer on first frame
        if self._writer is None:
            self._init_writer(width, height)

        # Render to image
        frame = self._render_to_image(glyph_string, width, height)

        # Write frame
        self._writer.write(frame)
        self._frame_count += 1

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape codes from text."""
        return self.ANSI_PATTERN.sub("", text)

    def _render_to_image(self, glyph_string: str, width: int, height: int) -> np.ndarray:
        """Render glyph string to image.

        Args:
            glyph_string: ANSI-colored string
            width: Width in characters
            height: Height in characters

        Returns:
            BGR image as numpy array
        """
        from PIL import Image, ImageDraw

        # Create image
        pixel_w = width * self.config.char_width
        pixel_h = height * self.config.char_height

        img = Image.new("RGB", (pixel_w, pixel_h), self.config.bg_color)
        draw = ImageDraw.Draw(img)

        # Parse and render
        lines = glyph_string.split("\n")
        current_fg = self.config.default_fg

        for y, line in enumerate(lines):
            x = 0
            pos = 0

            while pos < len(line):
                # Check for ANSI escape
                if line[pos : pos + 2] == "\033[":
                    # Find end of escape sequence
                    end = line.find("m", pos)
                    if end != -1:
                        codes = line[pos + 2 : end]
                        current_fg = self._parse_color(codes, current_fg)
                        pos = end + 1
                        continue

                # Regular character
                char = line[pos]

                # Draw character
                px = x * self.config.char_width
                py = y * self.config.char_height

                if self._font:
                    draw.text((px, py), char, font=self._font, fill=current_fg)
                else:
                    draw.text((px, py), char, fill=current_fg)

                x += 1
                pos += 1

        # Convert to numpy BGR
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for video recording.")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def render_to_image(self, glyph_string: str) -> np.ndarray:
        """Render a glyph string to a BGR image array."""
        if not isinstance(glyph_string, str):
            glyph_string = str(glyph_string)
        lines = glyph_string.split("\n")
        height = len(lines)
        width = max(len(self._strip_ansi(line)) for line in lines) if lines else 80
        return self._render_to_image(glyph_string, width, height)

    def _parse_color(self, codes: str, current: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Parse ANSI color codes.

        Args:
            codes: ANSI code string (e.g., "38;5;196" or "38;2;255;0;0")
            current: Current foreground color

        Returns:
            New foreground color
        """
        # Reset
        if codes == "0" or codes == "":
            return self.config.default_fg

        # TrueColor (24-bit)
        match = self.ANSI_TRUE_FG.search(codes)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))

        # ANSI256
        match = self.ANSI_256_FG.search(codes)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < 256:
                return self._ansi256_palette[idx]

        return current

    def close(self):
        """Close recorder and finalize video."""
        if self._writer:
            self._writer.release()
            self._writer = None
            print(f"✓ Recording saved: {self.config.output_path} ({self._frame_count} frames)")

    def mux_audio(self, audio_path: str) -> Optional[Path]:
        """Mux audio into the recorded video.

        Args:
            audio_path: Path to audio file

        Returns:
            Path to final output with audio, or None on failure
        """
        import subprocess

        if not self.config.output_path.exists():
            print(f"Video not found: {self.config.output_path}")
            return None

        if not Path(audio_path).exists():
            print(f"Audio not found: {audio_path}")
            return None

        # Temporary output
        temp_output = self.config.output_path.with_stem(
            self.config.output_path.stem + "_with_audio"
        )

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(self.config.output_path),
                "-i",
                audio_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                "-loglevel",
                "error",
                str(temp_output),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Replace original
                self.config.output_path.unlink()
                temp_output.rename(self.config.output_path)
                print(f"✓ Audio muxed: {self.config.output_path}")
                return self.config.output_path
            else:
                print(f"Audio mux failed: {result.stderr}")
                return None

        except Exception as e:
            print(f"Audio mux error: {e}")
            return None

    @property
    def frame_count(self) -> int:
        """Number of frames recorded."""
        return self._frame_count

    @property
    def is_open(self) -> bool:
        """Check if recorder is open."""
        return self._writer is not None

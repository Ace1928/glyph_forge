"""
High-Fidelity Glyph Renderer for Glyph Forge.

Converts video frames to beautiful terminal-displayable glyph art using:
- Gradient rendering (block characters)
- Vectorized operations (numpy)
- Cached lookup tables
- Multiple color modes (ANSI256, TrueColor)
"""

import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

try:  # Optional dependency (streaming extra)
    import cv2 as _cv2

    cv2: Any = _cv2
except Exception:  # pragma: no cover
    cv2 = None


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Character Sets
# ═══════════════════════════════════════════════════════════════

# Extended gradient - sorted by visual density
# Dense characters for dark pixels → sparse for bright (works on dark terminals)
GRADIENT_CHARS = "█▓▒░·˙ "

# Extended gradient with 500+ levels for maximum fidelity
EXTENDED_GRADIENT = "█▓▒░·˙ @%#*+=-:. MW&8%B@$ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:;`'\"^_-~|/\\()[]{}<>"

# ASCII-only fallback
ASCII_GRADIENT = "@%#*+=-:. "

# Braille patterns base (U+2800)
BRAILLE_BASE = 0x2800


# ═══════════════════════════════════════════════════════════════
# Lookup Tables
# ═══════════════════════════════════════════════════════════════


class LookupTables:
    """Pre-computed lookup tables for fast rendering.

    Tables are cached to disk for instant startup after first run.
    """

    _CACHE_NAME = "lookup_tables_v5.npz"
    _MAX_CACHE_BYTES = 2 * 1024 * 1024
    _REQUIRED_ARRAYS = {
        "luminance_to_char",
        "rgb_to_ansi256",
        "gradient_chars",
        "ansi256_palette_lab",
        "srgb_to_linear",
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize lookup tables.

        Args:
            cache_dir: Directory for cached tables
        """
        self.cache_dir = cache_dir or self._default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / self._CACHE_NAME

        self._luminance_to_char: Optional[np.ndarray] = None
        self._rgb_to_ansi256: Optional[np.ndarray] = None
        self._gradient_chars: Optional[np.ndarray] = None
        self._ansi256_palette_lab: Optional[np.ndarray] = None
        self._srgb_to_linear: Optional[np.ndarray] = None

        self._load_or_build()

    @staticmethod
    def _default_cache_dir() -> Path:
        """Return a platform-aware cache directory without creating it."""

        explicit = os.environ.get("GLYPH_FORGE_CACHE_DIR")
        if explicit:
            return Path(explicit).expanduser()
        if os.name == "nt":
            root = os.environ.get("LOCALAPPDATA")
            return (
                Path(root) / "GlyphForge" / "Cache"
                if root
                else Path.home() / ".glyph_forge" / "cache"
            )
        root = os.environ.get("XDG_CACHE_HOME")
        return (
            Path(root).expanduser() / "glyph-forge"
            if root
            else Path.home() / ".cache" / "glyph-forge"
        )

    def _load_or_build(self) -> None:
        """Load tables from cache or build them."""
        if self.cache_path.exists():
            try:
                self._load_cache()
                logger.debug("Loaded lookup tables from %s", self.cache_path)
                return
            except (OSError, ValueError, KeyError) as exc:
                logger.warning("Ignoring invalid Glyph Forge cache %s: %s", self.cache_path, exc)

        self._build_tables()
        self._save_cache()

    def _load_cache(self) -> None:
        """Load and validate a non-executable NumPy cache archive."""

        self._inspect_cache_archive()
        with np.load(self.cache_path, allow_pickle=False) as data:
            if not self._REQUIRED_ARRAYS.issubset(data.files):
                missing = sorted(self._REQUIRED_ARRAYS.difference(data.files))
                raise ValueError(f"cache is missing arrays: {', '.join(missing)}")
            luminance = data["luminance_to_char"]
            rgb = data["rgb_to_ansi256"]
            gradient = data["gradient_chars"]
            palette = data["ansi256_palette_lab"]
            srgb = data["srgb_to_linear"]

        if luminance.shape != (256,) or luminance.dtype.kind not in "ui":
            raise ValueError("invalid luminance lookup table")
        if rgb.shape != (32, 32, 32) or rgb.dtype.kind not in "ui":
            raise ValueError("invalid ANSI 256 lookup table")
        if gradient.ndim != 1 or not gradient.size or gradient.dtype.kind != "U":
            raise ValueError("invalid gradient table")
        if palette.shape != (256, 3) or palette.dtype.kind not in "f":
            raise ValueError("invalid ANSI 256 palette")
        if srgb.shape != (256,) or srgb.dtype.kind not in "f":
            raise ValueError("invalid sRGB lookup table")

        self._luminance_to_char = luminance
        self._rgb_to_ansi256 = rgb
        self._gradient_chars = gradient
        self._ansi256_palette_lab = palette
        self._srgb_to_linear = srgb

    def _inspect_cache_archive(self) -> None:
        """Reject oversized or structurally unexpected NPZ members before loading."""

        expected_names = {f"{name}.npy" for name in self._REQUIRED_ARRAYS}
        try:
            with zipfile.ZipFile(self.cache_path) as archive:
                members = archive.infolist()
                names = {member.filename for member in members}
                if names != expected_names:
                    raise ValueError("cache archive contains unexpected members")
                if sum(member.file_size for member in members) > self._MAX_CACHE_BYTES:
                    raise ValueError("cache archive expands beyond its safety limit")

                for member in members:
                    with archive.open(member) as stream:
                        version = np.lib.format.read_magic(stream)
                        if version == (1, 0):
                            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                        elif version == (2, 0):
                            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                        else:
                            raise ValueError(f"unsupported NumPy cache format: {version}")
                    self._validate_array_header(member.filename, shape, dtype)
        except zipfile.BadZipFile as exc:
            raise ValueError("cache is not a valid NPZ archive") from exc

    @staticmethod
    def _validate_array_header(
        filename: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> None:
        """Validate array dimensions before NumPy allocates their data."""

        if filename == "luminance_to_char.npy" and shape == (256,) and dtype.kind in "ui":
            return
        if filename == "rgb_to_ansi256.npy" and shape == (32, 32, 32) and dtype.kind in "ui":
            return
        if (
            filename == "gradient_chars.npy"
            and len(shape) == 1
            and 0 < shape[0] <= 1024
            and dtype.kind == "U"
        ):
            return
        if filename == "ansi256_palette_lab.npy" and shape == (256, 3) and dtype.kind == "f":
            return
        if filename == "srgb_to_linear.npy" and shape == (256,) and dtype.kind == "f":
            return
        raise ValueError(f"invalid array header for {filename}")

    def _build_tables(self):
        """Build all lookup tables."""
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for streaming rendering.")
        # Build density-sorted gradient characters
        chars = self._build_density_sorted_chars(EXTENDED_GRADIENT)
        self._gradient_chars = np.array(list(chars), dtype="U1")

        # Luminance → character (256 levels → char index)
        levels = np.linspace(0, len(chars) - 1, 256)
        self._luminance_to_char = levels.astype(np.uint16)

        # sRGB → linear LUT (0..255) for fast luminance
        srgb = np.linspace(0, 1, 256, dtype=np.float32)
        linear = np.where(
            srgb <= 0.04045,
            srgb / 12.92,
            ((srgb + 0.055) / 1.055) ** 2.4,
        )
        self._srgb_to_linear = linear.astype(np.float32)

        # RGB → ANSI256 (perceptual)
        self._build_ansi256_lab_palette()
        self._rgb_to_ansi256 = self._build_rgb_to_ansi256()

    def _compute_ansi256(self, r: int, g: int, b: int) -> int:
        """Compute closest ANSI256 color code."""
        # Check grayscale ramp (232-255)
        gray = (r + g + b) // 3
        if abs(r - gray) < 10 and abs(g - gray) < 10 and abs(b - gray) < 10:
            # Use grayscale
            gray_idx = int(gray / 255 * 23)
            return 232 + gray_idx

        # Use 6x6x6 color cube (16-231)
        r6 = int(r / 256 * 6)
        g6 = int(g / 256 * 6)
        b6 = int(b / 256 * 6)
        return 16 + 36 * r6 + 6 * g6 + b6

    def _save_cache(self) -> None:
        """Atomically save tables in a non-executable, portable archive."""

        if any(
            table is None
            for table in (
                self._luminance_to_char,
                self._rgb_to_ansi256,
                self._gradient_chars,
                self._ansi256_palette_lab,
                self._srgb_to_linear,
            )
        ):
            raise RuntimeError("lookup tables are incomplete")
        assert self._luminance_to_char is not None
        assert self._rgb_to_ansi256 is not None
        assert self._gradient_chars is not None
        assert self._ansi256_palette_lab is not None
        assert self._srgb_to_linear is not None

        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{self.cache_path.name}.",
                suffix=".tmp",
                dir=self.cache_dir,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.savez_compressed(
                    temporary,
                    luminance_to_char=self._luminance_to_char,
                    rgb_to_ansi256=self._rgb_to_ansi256,
                    gradient_chars=self._gradient_chars,
                    ansi256_palette_lab=self._ansi256_palette_lab,
                    srgb_to_linear=self._srgb_to_linear,
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.cache_path)
            if os.name != "nt":
                self.cache_path.chmod(0o600)
        except OSError as exc:
            logger.warning("Could not save Glyph Forge cache %s: %s", self.cache_path, exc)
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def get_char_index(self, luminance: np.ndarray) -> np.ndarray:
        """Map luminance values to character indices."""
        assert self._luminance_to_char is not None
        return self._luminance_to_char[luminance]

    def get_ansi256(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Map RGB values to ANSI256 codes."""
        r5 = (r >> 3).astype(np.uint8)
        g5 = (g >> 3).astype(np.uint8)
        b5 = (b >> 3).astype(np.uint8)
        assert self._rgb_to_ansi256 is not None
        return self._rgb_to_ansi256[r5, g5, b5]

    def get_gradient_chars(self) -> np.ndarray:
        assert self._gradient_chars is not None
        return self._gradient_chars

    def get_srgb_to_linear(self) -> np.ndarray:
        assert self._srgb_to_linear is not None
        return self._srgb_to_linear

    def _build_density_sorted_chars(self, chars: str) -> str:
        font = self._load_font()
        densities = []
        for ch in chars:
            density = self._char_density(ch, font)
            densities.append((density, ch))
        densities.sort(reverse=True)  # darkest first
        return "".join(ch for _, ch in densities)

    def _load_font(self):
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "C:\\Windows\\Fonts\\consola.ttf",
        ]
        for path in font_paths:
            if Path(path).exists():
                return ImageFont.truetype(path, 14)
        return ImageFont.load_default()

    def _char_density(self, ch: str, font) -> float:
        img = Image.new("L", (20, 20), color=255)
        draw = ImageDraw.Draw(img)
        draw.text((2, 2), ch, font=font, fill=0)
        arr = np.array(img)
        return 1.0 - (arr.mean() / 255.0)

    def _build_ansi256_lab_palette(self):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for ANSI256 lookup tables.")
        palette = []
        # Standard 16 colors
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
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    R = 0 if r == 0 else 55 + r * 40
                    G = 0 if g == 0 else 55 + g * 40
                    B = 0 if b == 0 else 55 + b * 40
                    palette.append((R, G, B))
        for i in range(24):
            gray = 8 + i * 10
            palette.append((gray, gray, gray))
        palette_arr = np.array(palette, dtype=np.uint8).reshape((-1, 1, 3))
        lab = cv2.cvtColor(palette_arr, cv2.COLOR_RGB2LAB).reshape((-1, 3))
        self._ansi256_palette_lab = lab.astype(np.float32)

    def _nearest_ansi256_lab(self, r: int, g: int, b: int) -> int:
        rgb = np.array([[[r, g, b]]], dtype=np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape((3,)).astype(np.float32)
        diff = self._ansi256_palette_lab - lab
        dist = np.sum(diff * diff, axis=1)
        return int(np.argmin(dist))

    def _build_rgb_to_ansi256(self) -> np.ndarray:
        """Vectorized RGB(5bit) -> ANSI256 lookup."""
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for ANSI256 lookup tables.")
        # Build 32^3 RGB grid
        vals: NDArray[np.uint8] = np.arange(32, dtype=np.uint8)
        grids = cast(
            tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]],
            np.meshgrid(vals, vals, vals, indexing="ij"),
        )
        r: NDArray[np.uint8] = grids[0]
        g: NDArray[np.uint8] = grids[1]
        b: NDArray[np.uint8] = grids[2]
        rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
        rgb = (rgb * (255.0 / 31.0)).reshape((-1, 1, 3)).astype(np.uint8)
        # Convert to Lab in one batch
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape((-1, 3)).astype(np.float32)
        # Find nearest palette entries in chunks to cap memory
        palette = self._ansi256_palette_lab
        assert palette is not None
        chunk = 4096
        out = np.empty((lab.shape[0],), dtype=np.uint8)
        for i in range(0, lab.shape[0], chunk):
            block = lab[i : i + chunk]
            diff = block[:, None, :] - palette[None, :, :]
            dist = np.sum(diff * diff, axis=2)
            out[i : i + chunk] = np.argmin(dist, axis=1).astype(np.uint8)
        return out.reshape((32, 32, 32))


# ═══════════════════════════════════════════════════════════════
# Renderer
# ═══════════════════════════════════════════════════════════════


@dataclass
class RenderConfig:
    """Renderer configuration."""

    mode: str = "gradient"  # gradient, braille, ascii, blocks
    color: str = "ansi256"  # none, ansi256, truecolor
    dithering: bool = False
    gamma: float = 1.15
    contrast: float = 0.98
    brightness: float = 0.02
    auto_contrast: bool = True
    cache_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        """Normalize and validate public rendering options."""

        self.mode = self.mode.lower()
        self.color = self.color.lower()
        if self.mode not in {"ascii", "blocks", "braille", "gradient", "hybrid"}:
            raise ValueError(f"unsupported render mode: {self.mode}")
        if self.color not in {"ansi256", "none", "truecolor"}:
            raise ValueError(f"unsupported color mode: {self.color}")
        if self.gamma <= 0:
            raise ValueError("gamma must be greater than zero")
        if self.contrast < 0:
            raise ValueError("contrast cannot be negative")


class GlyphRenderer:
    """High-performance glyph renderer.

    Converts video frames to terminal-displayable glyph strings.
    Uses vectorized numpy operations and cached lookup tables.

    Usage:
        renderer = GlyphRenderer()
        glyph_string = renderer.render(frame, width=120, height=40)
        print(glyph_string)
    """

    def __init__(self, config: Optional[RenderConfig] = None):
        """Initialize renderer.

        Args:
            config: Render configuration
        """
        self.config = config or RenderConfig()
        self.tables = LookupTables(self.config.cache_dir)

        # Pre-compute character array for fast lookup
        self._char_array = self.tables.get_gradient_chars()
        self._ascii_char_array = np.array(list(ASCII_GRADIENT), dtype="U1")
        self._block_char_array = np.array(list(GRADIENT_CHARS), dtype="U1")
        self._srgb_to_linear = self.tables.get_srgb_to_linear()
        self._tone_lut = self._build_tone_lut()
        self._bayer = (
            np.array(
                [
                    [0, 8, 2, 10],
                    [12, 4, 14, 6],
                    [3, 11, 1, 9],
                    [15, 7, 13, 5],
                ],
                dtype=np.float32,
            )
            / 16.0
        ) - 0.5
        self._braille_chars = np.array([chr(BRAILLE_BASE + i) for i in range(256)], dtype="U1")
        self._braille_weights = np.array(
            [
                [0x01, 0x08],  # Row 0: dots 0, 3
                [0x02, 0x10],  # Row 1: dots 1, 4
                [0x04, 0x20],  # Row 2: dots 2, 5
                [0x40, 0x80],  # Row 3: dots 6, 7
            ],
            dtype=np.uint8,
        )

        # Color code cache
        self._color_cache: Dict[int, str] = {}

    def render(
        self,
        frame: np.ndarray,
        width: int = 120,
        height: int = 40,
        color: Optional[str] = None,
    ) -> str:
        """Render frame to glyph string.

        Args:
            frame: BGR image (numpy array)
            width: Output width in characters
            height: Output height in characters
            color: Color mode override ('none', 'ansi256', 'truecolor')

        Returns:
            Terminal-ready string with ANSI escape codes
        """
        if width < 1 or height < 1:
            raise ValueError("render dimensions must be positive")
        if width > 8192 or height > 8192:
            raise ValueError("render dimensions cannot exceed 8192 characters")
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array")
        if frame.ndim not in {2, 3} or (frame.ndim == 3 and frame.shape[2] not in {1, 3, 4}):
            raise ValueError("frame must be grayscale, BGR, or BGRA image data")
        if frame.size == 0:
            raise ValueError("frame cannot be empty")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame[:, :, 0]

        color = (color or self.config.color).lower()
        if color not in {"ansi256", "none", "truecolor"}:
            raise ValueError(f"unsupported color mode: {color}")

        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for streaming rendering.")
        if self.config.mode == "braille":
            return self.render_braille(frame, width=width, height=height)
        # Resize frame to target dimensions
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # Convert to perceptual luminance for character mapping
        gray = self._compute_luminance(frame)
        gray = self._tone_map(gray)
        if self.config.dithering:
            gray = self._ordered_dither(gray)

        # Get character indices and the selected portable character palette.
        char_indices: np.ndarray
        if self.config.mode == "ascii":
            self._char_array = self._ascii_char_array
            char_indices = (gray.astype(np.uint16) * (len(self._char_array) - 1) // 255).astype(
                np.uint8
            )
        elif self.config.mode == "blocks":
            self._char_array = self._block_char_array
            char_indices = (gray.astype(np.uint16) * (len(self._char_array) - 1) // 255).astype(
                np.uint8
            )
        else:
            self._char_array = self.tables.get_gradient_chars()
            char_indices = self.tables.get_char_index(gray)

        # Build output based on color mode
        if color == "none":
            return self._render_no_color(char_indices)
        if color == "ansi256":
            return self._render_ansi256(frame, char_indices)
        if color == "truecolor":
            return self._render_truecolor(frame, char_indices)
        raise AssertionError("validated color mode reached an unreachable branch")

    def _render_no_color(self, char_indices: np.ndarray) -> str:
        """Render without color (fastest)."""
        chars = self._char_array[char_indices]

        lines = []
        for row in chars:
            lines.append("".join(row))

        return "\n".join(lines)

    def _render_ansi256(self, frame: np.ndarray, char_indices: np.ndarray) -> str:
        """Render with ANSI256 colors (good balance)."""
        # Get color codes
        if len(frame.shape) == 3:
            b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        else:
            r = g = b = frame

        color_codes = self.tables.get_ansi256(r, g, b)
        chars = self._char_array[char_indices]

        # Build output with run-length encoding
        lines = []

        for y in range(frame.shape[0]):
            line_parts = []
            last_code = -1  # Reset for each line

            for x in range(frame.shape[1]):
                code = int(color_codes[y, x])
                char = chars[y, x]

                if code != last_code:
                    line_parts.append(f"\033[38;5;{code}m{char}")
                    last_code = code
                else:
                    line_parts.append(char)

            lines.append("".join(line_parts))

        return "\033[0m" + "\n".join(lines) + "\033[0m"

    def _render_truecolor(self, frame: np.ndarray, char_indices: np.ndarray) -> str:
        """Render with 24-bit TrueColor (highest quality)."""
        if len(frame.shape) == 3:
            b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        else:
            r = g = b = frame

        chars = self._char_array[char_indices]

        lines = []
        for y in range(frame.shape[0]):
            line_parts = []
            last_rgb = (-1, -1, -1)

            for x in range(frame.shape[1]):
                R, G, B = int(r[y, x]), int(g[y, x]), int(b[y, x])
                char = chars[y, x]

                if (R, G, B) != last_rgb:
                    line_parts.append(f"\033[38;2;{R};{G};{B}m{char}")
                    last_rgb = (R, G, B)
                else:
                    line_parts.append(char)

            lines.append("".join(line_parts))

        return "\033[0m" + "\n".join(lines) + "\033[0m"

    def _compute_luminance(self, frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 2:
            return frame.astype(np.uint8)
        # BGR -> RGB
        b = frame[:, :, 0]
        g = frame[:, :, 1]
        r = frame[:, :, 2]
        # sRGB to linear via LUT
        r_lin = self._srgb_to_linear[r]
        g_lin = self._srgb_to_linear[g]
        b_lin = self._srgb_to_linear[b]
        # luminance
        lum = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
        return np.clip(lum * 255.0, 0, 255).astype(np.uint8)

    def _tone_map(self, gray: np.ndarray) -> np.ndarray:
        if self.config.auto_contrast:
            if cv2 is None:
                raise RuntimeError("OpenCV (cv2) is required for auto-contrast.")
            gray = cv2.equalizeHist(gray)
        return self._tone_lut[gray]

    def _build_tone_lut(self) -> np.ndarray:
        g = np.linspace(0, 1, 256, dtype=np.float32)
        g = np.clip((g - 0.5) * self.config.contrast + 0.5 + self.config.brightness, 0, 1)
        g = np.power(g, 1.0 / max(0.01, self.config.gamma))
        return np.clip(g * 255.0, 0, 255).astype(np.uint8)

    def _ordered_dither(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        tile = np.tile(self._bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]
        g = gray.astype(np.float32) / 255.0
        g = np.clip(g + tile * 0.08, 0, 1)
        return (g * 255.0).astype(np.uint8)

    def render_braille(
        self,
        frame: np.ndarray,
        width: int = 120,
        height: int = 40,
        threshold: int = 128,
    ) -> str:
        """Render using Braille characters (2x4 sub-pixel resolution).

        Each Braille character represents a 2x4 pixel block, giving
        effectively 2x vertical and 2x horizontal resolution.

        Args:
            frame: BGR image
            width: Output width in characters
            height: Output height in characters
            threshold: Binary threshold for dots

        Returns:
            Braille-rendered string
        """
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for braille rendering.")
        # Braille uses 2x4 blocks, so we need 2x width and 4x height pixels
        pixel_w = width * 2
        pixel_h = height * 4

        # Resize and convert to grayscale
        frame = cv2.resize(frame, (pixel_w, pixel_h), interpolation=cv2.INTER_AREA)
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        gray_norm = gray.astype(np.float32) / 255.0

        # Vectorized 2x4 blocks: (height, width, 4, 2)
        h, w = gray_norm.shape
        h_blocks = h // 4
        w_blocks = w // 2
        block = gray_norm[: h_blocks * 4, : w_blocks * 2].reshape(h_blocks, 4, w_blocks, 2)
        block = block.transpose(0, 2, 1, 3)  # (h_blocks, w_blocks, 4, 2)

        # Adaptive threshold per block (mean-based)
        mean = block.mean(axis=(2, 3), keepdims=True)
        mask = block < mean  # darker pixels become dots

        # Braille dot positions (column-major):
        # 0 3
        # 1 4
        # 2 5
        # 6 7
        codes = (mask * self._braille_weights[None, None, :, :]).sum(axis=(2, 3)).astype(np.uint8)
        chars = self._braille_chars[codes]

        return "\n".join("".join(row) for row in chars)

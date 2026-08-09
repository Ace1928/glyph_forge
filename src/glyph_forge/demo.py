"""Self-contained Glyph Forge showcase: memes, media, modes, fonts, and styles.

The demo acquires its own assets.  Meme templates and video thumbnails are
downloaded from public endpoints into a small cache, and every asset falls
back to procedurally generated art when the network is unavailable or a
source refuses a request, so the show always completes.

Network behaviour is controlled with ``--offline`` (never touch the network)
and ``--media`` (fetch video thumbnails).  The test suite points the demo at
a local HTTP server through the ``GLYPH_FORGE_DEMO_MEME_BASE`` and
``GLYPH_FORGE_DEMO_THUMB_BASE`` environment variables.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

from .core.style_manager import apply_style, get_available_styles
from .live.renderers import (
    FrameRenderer,
    RenderConfig,
    RenderMode,
    normalize_render_mode,
)
from .runtime import detect_runtime_profile
from .services.text_to_banner import text_to_banner
from .utils.alphabet_manager import AlphabetManager

_MEME_BASE = os.environ.get("GLYPH_FORGE_DEMO_MEME_BASE", "https://api.memegen.link")
_THUMB_BASE = os.environ.get(
    "GLYPH_FORGE_DEMO_THUMB_BASE", "https://img.youtube.com/vi"
)
_CACHE_ROOT = (
    Path(os.environ.get("GLYPH_FORGE_DEMO_CACHE", str(Path.home() / ".cache")))
    / "glyph_forge"
    / "demo"
)

_FETCH_TIMEOUT = 6.0
_FETCH_RETRIES = 2
_UA = (
    "Mozilla/5.0 (compatible; glyph-forge-demo/0.2; +https://github.com/"
    "Ace1928/glyph_forge)"
)

_MEMES: Tuple[Tuple[str, str, Sequence[str]], ...] = (
    (
        "drake",
        "me installing glyph-forge",
        ("me after one weekend with glyph-forge",),
    ),
    (
        "distracted",
        "my todo list",
        ("glyph-forge demo", "git commit -m 'art'"),
    ),
    (
        "crying-floor",
        "it rendered",
        (),
    ),
)

_VIDEO_IDS: Tuple[Tuple[str, str], ...] = (
    ("dQw4w9WgXcQ", "never gonna give you up (the internet's favourite prank)"),
    ("jNQXAC9IVRw", "the very first YouTube video, 'Me at the zoo'"),
)


@dataclass(frozen=True, slots=True)
class DemoArtifact:
    """One file written by a demo run."""

    name: str
    path: Path
    kind: str


@dataclass(frozen=True, slots=True)
class DemoStats:
    """Counts and timings collected while the show ran."""

    scenes: int
    renders: int
    fonts_shown: int
    styles_shown: int
    assets_fetched: int
    assets_fallback: int
    elapsed_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenes": self.scenes,
            "renders": self.renders,
            "fonts_shown": self.fonts_shown,
            "styles_shown": self.styles_shown,
            "assets_fetched": self.assets_fetched,
            "assets_fallback": self.assets_fallback,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


@dataclass(frozen=True, slots=True)
class DemoResult:
    """The complete narrated showcase plus everything it wrote to disk."""

    text: str
    artifacts: Tuple[DemoArtifact, ...]
    stats: DemoStats

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "artifacts": [
                {"name": item.name, "path": str(item.path), "kind": item.kind}
                for item in self.artifacts
            ],
            "stats": self.stats.to_dict(),
        }


class DemoBuilder:
    """Accumulates narrated scenes and writes optional artifact files."""

    def __init__(
        self,
        *,
        color: bool,
        offline: bool,
        media: bool,
        mode: Optional[str],
        width: Optional[int],
        performance: str,
        output_dir: Optional[Path],
    ) -> None:
        self.color = color
        self.offline = offline
        self.media = media
        self.mode = mode
        self.width = width
        self.performance = performance
        self.output_dir = output_dir
        self.parts: List[str] = []
        self.artifacts: List[DemoArtifact] = []
        self.renders = 0
        self.fetched = 0
        self.fallback = 0
        self.profile = detect_runtime_profile(performance)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

    # -- helpers ------------------------------------------------------------

    @property
    def selected_modes(self) -> List[RenderMode]:
        modes: List[RenderMode]
        if self.mode in (None, "all"):
            modes = list(RenderMode)
        else:
            modes = [cast(RenderMode, normalize_render_mode(self.mode))]
        return modes

    def _grid_width(self) -> int:
        if self.width is not None:
            return max(10, self.width)
        columns = shutil.get_terminal_size((self.profile.stream_width, 24)).columns
        if sys.stdout.isatty():
            return max(20, columns - 2)
        return self.profile.stream_width

    def _say(self, text: str) -> None:
        self.parts.append(text)

    def _scene(self, label: str) -> None:
        self.parts.append(f"\n━━━═◇  {label}  ◇═━━━\n")

    def _save(self, name: str, content: str) -> None:
        if self.output_dir is None:
            return
        path = self.output_dir / name
        path.write_text(content, encoding="utf-8")
        self.artifacts.append(DemoArtifact(name=name, path=path, kind="text"))

    def _fetch(self, url: str) -> Optional[bytes]:
        if self.offline:
            return None
        cached_root = _CACHE_ROOT
        cached = cached_root / (_cache_key(url) + ".bin")
        if cached.is_file():
            return cached.read_bytes()
        for _ in range(_FETCH_RETRIES):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": _UA})
                with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT) as resp:
                    raw: bytes = resp.read()
                if len(raw) > 4 * 1024 * 1024:
                    return None
                cached_root.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(raw)
                return raw
            except (urllib.error.URLError, OSError, TimeoutError):
                continue
        return None

    def _render_pixels(self, pixels: np.ndarray, mode: RenderMode, width: int) -> str:
        color = (
            "truecolor" if (self.color and mode is RenderMode.HALF_BLOCK) else "none"
        )
        renderer = FrameRenderer(
            RenderConfig(
                width=width,
                mode=mode,
                color=color,
                charset="detailed",
                edge_algorithm="sobel",
                resample=self.profile.resample,
            )
        )
        rendered = renderer.render(pixels)
        self.renders += 1
        return rendered.text

    def _save_image(self, name: str, image: Any) -> None:
        if self.output_dir is None:
            return
        path = self.output_dir / name
        image.convert("RGB").save(path, format="PNG", optimize=True)
        self.artifacts.append(DemoArtifact(name=name, path=path, kind="image"))

    # -- asset acquisition ----------------------------------------------------

    def _meme_image(self, template: str, top: str, lower: Sequence[str]) -> Any:
        """Return meme pixels; procedural art replaces any failure."""
        parts = [template, top, *(lower or ())]
        url = (
            _MEME_BASE
            + "/images/"
            + "/".join(urllib.parse.quote(part.replace(" ", "_")) for part in parts)
            + ".png"
        )
        payload = self._fetch(url)
        if payload is not None:
            self.fetched += 1
            try:
                from PIL import Image

                return Image.open(io.BytesIO(payload)).convert("RGB")
            except OSError:  # pragma: no cover - resilient against junk responses
                self.fetched -= 1
        self.fallback += 1
        return self._procedural_meme(top, lower)

    def _thumbnail_image(self, video_id: str) -> Any:
        url = f"{_THUMB_BASE}/{video_id}/maxresdefault.jpg"
        payload = self._fetch(url)
        if payload is not None:
            self.fetched += 1
            try:
                from PIL import Image

                return Image.open(io.BytesIO(payload)).convert("RGB")
            except OSError:  # pragma: no cover - resilient against junk responses
                self.fetched -= 1
        self.fallback += 1
        return self._procedural_thumbnail(video_id)

    def _procedural_meme(self, top: str, lower: Sequence[str]) -> Any:
        """A deterministic offline meme with the same two-line layout."""
        from PIL import Image, ImageDraw, ImageFont

        seed_source = top + "/" + "/".join(lower)
        rng = np.random.default_rng(sum(ord(char) for char in seed_source))
        size = (640, 360)
        y, x = np.indices(size, dtype=np.float32)
        angle = float(rng.uniform(0.02, 0.35))
        channels = [
            (
                np.clip(np.sin(angle * (x * 4 + y * 4) + phase * 3) * 0.5 + 0.5, 0, 1)
                * 255
            ).astype(np.uint8)
            for phase in rng.uniform(0, 6.28, 3)
        ]
        image = Image.fromarray(np.dstack(channels).astype(np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.rectangle((0, 0, 640, 40), fill="black")
        draw.rectangle((0, 320, 640, 360), fill="black")
        draw.text((16, 8), top, fill=(255, 255, 255), font=font)
        draw.text((16, 328), " · ".join(lower), fill=(255, 255, 255), font=font)
        return image

    def _procedural_thumbnail(self, video_id: str) -> Any:
        """Offline stand-in: a video test-card with the id burned in."""
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (1280, 720), (14, 20, 30))
        draw = ImageDraw.Draw(image)
        for y in range(0, 720, 8):
            draw.rectangle((0, y, 1280, y + 4), fill=(22, 32, 48))
        draw.rectangle((440, 260, 840, 460), outline=(240, 200, 60), width=6)
        draw.ellipse((560, 300, 720, 440), fill=(240, 200, 60))
        draw.polygon((720, 320, 720, 420, 800, 370), fill=(240, 200, 60))
        draw.text((540, 500), video_id[:8], fill=(255, 255, 255))
        return image

    # -- scenes ----------------------------------------------------------------

    def scene_opening(self) -> None:
        self._scene("GLYPH FORGE · the whole toolkit, one self-contained show")
        width = min(self._grid_width(), 90)
        for font in ("slant", "small", "banner3"):
            self._say(
                text_to_banner(
                    "GLYPH FORGE",
                    font=font,
                    style="minimal",
                    width=width * 2,
                )
            )
        self._say(
            "Everything you are about to see was rendered by Glyph Forge itself: "
            "no images, fonts, or files were shipped in the package."
        )

    def scene_charsets(self) -> int:
        self._scene("CHARACTER SETS · the density ramps")
        for name in ("general", "detailed", "block", "minimal", "eidosian"):
            alphabet = AlphabetManager.get_alphabet(name)
            self._say(f"[{name}] {alphabet[:56]}")
        return 5

    def scene_styles(self) -> int:
        self._scene("TEXT STYLES · one phrase, every mood")
        phrase = "EAT SLEEP GLYPH REPEAT"
        count = 0
        for name, data in sorted(get_available_styles().items()):
            if name in {"minimal", "boxed"}:
                continue
            styled = apply_style(text_to_banner(phrase, font="small", width=80), name)
            self._say(f"\nstyle [{name}] — {str(data.get('description', ''))}")
            self._say(styled)
            count += 1
        return count

    def scene_memes(self) -> None:
        label = "OFFLINE MODE" if self.offline else "downloaded live"
        self._scene(f"MEME WALL · public templates, {label}")
        for template, top, lower in _MEMES:
            image = self._meme_image(template, top, lower)
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            self._say(f"\n▶ {template.replace('-', ' ')} — '{top}'")
            self._save_image(f"meme-{template}.png", image)
            for mode in self.selected_modes:
                grid = self._render_pixels(pixels, mode, self._grid_width())
                self._say(f"  Glyph Forge · {mode.value}")
                self._say(grid)
                self._save(f"meme-{template}.{mode.value}.txt", grid)

    def scene_modes(self) -> None:
        self._scene("WHERE IT LIVES · the same image, every render mode")
        from .benchmark import synthetic_frame

        pixels = synthetic_frame(480, 270)
        for mode in self.selected_modes:
            grid = self._render_pixels(pixels, mode, self._grid_width())
            self._say(f"\n  Glyph Forge · {mode.value}")
            self._say(grid)
            self._save(f"mode-{mode.value}.txt", grid)
            if mode is RenderMode.HALF_BLOCK and self.color:
                self._say("  (half-block renders true colour when the terminal allows)")

    def scene_media(self) -> None:
        if not self.media:
            self._say(
                "\n[media skipped with --no-media]\n"
                "  glyph-forge live url 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'"
            )
            return
        self._scene("VIDEO · the internet's favourite clips, frame one")
        for video_id, caption in _VIDEO_IDS:
            thumbnail = self._thumbnail_image(video_id)
            pixels = np.asarray(thumbnail.convert("RGB"), dtype=np.uint8)
            self._say(f"\n▶ {caption}")
            for mode in self.selected_modes[:2]:
                grid = self._render_pixels(pixels, mode, self._grid_width())
                self._say(f"  Glyph Forge · {mode.value}")
                self._say(grid)
                self._save(f"video-{video_id}.{mode.value}.txt", grid)
            if self.offline:
                self._say("  (offline stand-in rendered instead of the real thumbnail)")
        self._say(
            "\nWant the whole thing moving? The same clips play live in your terminal:"
        )
        self._say(
            "  glyph-forge live url 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'"
        )
        self._say("  glyph-forge live video clip.mp4")

    def scene_outro(self, started: float) -> None:
        elapsed = time.perf_counter() - started
        self._scene("THAT'S THE WHOLE TOOLKIT")
        self._say(
            text_to_banner(
                "FORGE ON",
                font="slant",
                style="minimal",
                width=min(self._grid_width(), 90),
            )
        )
        modes = "all modes" if self.mode in (None, "all") else self.mode
        self._say(
            f"{modes} · {self.renders} renders · {self.fetched} assets fetched · "
            f"{self.fallback} crafted offline · {elapsed:.1f}s"
        )
        self._say("Next moves:")
        self._say("  glyph-forge studio      # drag, drop, and play in the browser")
        self._say("  glyph-forge benchmark    # measure this machine")
        self._say("  glyph-forge doctor       # verify every optional feature")
        self._say("  glyph-forge share art.mp4 --lan  # send it to friends")

    # -- driving ----------------------------------------------------------------

    def build(self) -> DemoResult:
        started = time.perf_counter()
        fonts = 3
        self.scene_opening()
        styles = self.scene_charsets()
        styles += self.scene_styles()
        self.scene_memes()
        self.scene_modes()
        self.scene_media()
        self.scene_outro(started)
        stats = DemoStats(
            scenes=6,
            renders=self.renders,
            fonts_shown=fonts,
            styles_shown=styles,
            assets_fetched=self.fetched,
            assets_fallback=self.fallback,
            elapsed_seconds=time.perf_counter() - started,
        )
        return DemoResult(
            text="\n".join(self.parts) + "\n",
            artifacts=tuple(self.artifacts),
            stats=stats,
        )


def run_demo(
    *,
    mode: Optional[str] = None,
    width: Optional[int] = None,
    color: bool = True,
    offline: bool = False,
    media: bool = True,
    performance: str = "auto",
    output_dir: Optional[Path] = None,
) -> DemoResult:
    """Run the full self-contained showcase and return its text and files."""
    return DemoBuilder(
        color=color,
        offline=offline,
        media=media,
        mode=mode,
        width=width,
        performance=performance,
        output_dir=output_dir,
    ).build()


def _cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


__all__ = [
    "DemoArtifact",
    "DemoResult",
    "DemoStats",
    "run_demo",
]

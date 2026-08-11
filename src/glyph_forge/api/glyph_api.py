"""High-level, typed Python API for Glyph Forge."""

from __future__ import annotations

import logging
import os
import threading
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, cast

from ..config.settings import ConfigManager, get_config
from ..contracts import RenderArtifact, RenderRequest
from ..core.banner_generator import BannerGenerator
from ..core.style_manager import get_available_styles
from ..persistence import AtomicWriteError, atomic_write_text
from ..rendering import ImageSource, format_for_path, render_image
from ..utils.alphabet_manager import AlphabetManager
from ..visual_defaults import DEFAULT_BRIGHTNESS, DEFAULT_CONTRAST

if TYPE_CHECKING:
    from ..batch import BatchRenderReport
    from ..projects import GlyphProject, ProjectSession, RenderPreset

logger = logging.getLogger(__name__)


class GlyphForgeAPI:
    """Stable application API backed by the canonical rendering pipeline."""

    def __init__(self, config: ConfigManager | None = None) -> None:
        self.config = config or get_config()
        default_font = self.config.get("banner", "default_font", "slant")
        default_width = self.config.get("banner", "default_width", 80)
        self._banner_generator = BannerGenerator(
            font=str(default_font),
            width=int(default_width),
        )
        logger.debug(
            "Glyph Forge API initialized with font=%r, width=%d",
            default_font,
            default_width,
        )

    def generate_banner(
        self,
        text: str,
        style: Optional[str] = None,
        font: Optional[str] = None,
        width: Optional[int] = None,
        effects: Optional[List[str]] = None,
        color: bool = False,
    ) -> str:
        """Generate a styled FIGlet banner."""

        selected_style = style or str(
            self.config.get("banner", "default_style", "minimal")
        )
        if font is not None or width is not None:
            selected_font = font if font is not None else self._banner_generator.font
            selected_width = (
                width if width is not None else self._banner_generator.width
            )
            generator = BannerGenerator(font=selected_font, width=selected_width)
            return cast(
                str,
                generator.generate(
                    text,
                    style=selected_style,
                    effects=effects,
                    color=color,
                ),
            )
        return cast(
            str,
            self._banner_generator.generate(
                text,
                style=selected_style,
                effects=effects,
                color=color,
            ),
        )

    def render_image(
        self,
        source: ImageSource,
        request: RenderRequest | None = None,
        *,
        destination: str | os.PathLike[str] | None = None,
    ) -> RenderArtifact:
        """Render an image into a structured artifact.

        Unlike the deprecated stateful converter, this method raises typed
        ``GlyphForgeRenderError`` subclasses when loading, rendering, encoding,
        or saving fails.
        """

        selected = request or RenderRequest(
            width=int(self.config.get("image", "default_width", 100)),
            charset=str(self.config.get("image", "default_charset", "general")),
            brightness=float(
                self.config.get("image", "brightness", DEFAULT_BRIGHTNESS)
            ),
            contrast=float(self.config.get("image", "contrast", DEFAULT_CONTRAST)),
            dither=bool(self.config.get("image", "dithering", False)),
        )
        return render_image(source, selected, destination=destination)

    def create_project(
        self,
        project_path: str | os.PathLike[str],
        source: str | os.PathLike[str],
        *,
        name: str | None = None,
        request: RenderRequest | None = None,
    ) -> "GlyphProject":
        """Create a portable project for an asset inside its project directory."""

        from ..projects import (
            AssetReference,
            GlyphProject,
            RecentProjectStore,
            save_project,
        )

        destination = Path(project_path).expanduser()
        project = GlyphProject.create(
            name or destination.name.removesuffix(".glyphforge.json") or "Untitled",
            AssetReference.from_path(source, destination),
            request,
        )
        save_project(project, destination)
        RecentProjectStore().touch(destination)
        return project

    def open_project(
        self,
        project_path: str | os.PathLike[str],
        *,
        recover: bool = True,
        autosave_delay: float | None = 1.0,
    ) -> "ProjectSession":
        """Open a project session with recovery, history, and autosave."""

        from ..projects import ProjectSession, RecentProjectStore

        session = ProjectSession.open(
            project_path,
            recover=recover,
            autosave_delay=autosave_delay,
        )
        RecentProjectStore().touch(project_path)
        return session

    def render_project(
        self,
        project_path: str | os.PathLike[str],
        *,
        destination: str | os.PathLike[str] | None = None,
        variant: str | None = None,
    ) -> RenderArtifact:
        """Render one project variant through the canonical still pipeline."""

        from ..projects import ProjectValidationError, load_project

        path = Path(project_path).expanduser()
        project = load_project(path)
        selected = project.active
        if variant is not None:
            matches = [
                item
                for item in project.variants
                if item.identifier == variant.casefold()
            ]
            if not matches:
                raise ProjectValidationError(f"unknown variant {variant!r}")
            selected = matches[0]
        return render_image(
            project.source.resolve(path),
            selected.request,
            destination=destination,
        )

    def render_preset(
        self,
        preset_path: str | os.PathLike[str],
        source: ImageSource,
        *,
        destination: str | os.PathLike[str] | None = None,
    ) -> RenderArtifact:
        """Render an image with a portable preset document."""

        from ..projects import load_preset

        return render_image(
            source,
            load_preset(preset_path).request,
            destination=destination,
        )

    def render_batch(
        self,
        sources: Iterable[str | os.PathLike[str]],
        output_directory: str | os.PathLike[str],
        preset: "RenderPreset | str | os.PathLike[str]",
        *,
        workers: int = 1,
        fail_fast: bool = False,
    ) -> "BatchRenderReport":
        """Render a bounded still queue from one shared preset."""

        from ..batch import items_for_sources, render_batch
        from ..projects import RenderPreset, load_preset

        selected = preset if isinstance(preset, RenderPreset) else load_preset(preset)
        return render_batch(
            items_for_sources(sources, output_directory, selected.request),
            workers=workers,
            fail_fast=fail_fast,
        )

    def image_to_glyph(
        self,
        image_path: ImageSource,
        output_path: str | os.PathLike[str] | None = None,
        charset: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        invert: bool = False,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        dithering: bool = False,
        color_mode: str = "none",
        mode: str = "glyph",
        style: str | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
        fit: str = "contain",
        alignment: str = "center",
        foreground: str = "#e8fff7",
        background: str = "#07110f",
    ) -> str:
        """Convenience image conversion returning text while optionally saving.

        Use :meth:`render_image` when callers need binary PNG data, render
        metrics, exact geometry, or the serialized request.
        """

        destination = Path(output_path) if output_path is not None else None
        request = RenderRequest(
            width=(
                width
                if width is not None
                else int(self.config.get("image", "default_width", 100))
            ),
            height=height,
            charset=(
                charset
                if charset is not None
                else str(self.config.get("image", "default_charset", "general"))
            ),
            mode=mode,
            output_format=format_for_path(destination, color=color_mode),
            invert=invert,
            dither=dithering,
            brightness=(
                brightness
                if brightness is not None
                else float(self.config.get("image", "brightness", DEFAULT_BRIGHTNESS))
            ),
            contrast=(
                contrast
                if contrast is not None
                else float(self.config.get("image", "contrast", DEFAULT_CONTRAST))
            ),
            style=style,
            output_width=output_width,
            output_height=output_height,
            fit=fit,
            alignment=alignment,
            foreground=foreground,
            background=background,
        )
        artifact = self.render_image(image_path, request, destination=destination)
        return artifact.data if isinstance(artifact.data, str) else artifact.glyph_text

    def image_to_Glyph(
        self,
        image_path: ImageSource,
        output_path: str | os.PathLike[str] | None = None,
        charset: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        invert: bool = False,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        dithering: bool = False,
        color_mode: str = "none",
        **options: Any,
    ) -> str:
        """Deprecated mixed-case alias for :meth:`image_to_glyph`."""

        warnings.warn(
            "GlyphForgeAPI.image_to_Glyph is deprecated; use image_to_glyph",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.image_to_glyph(
            image_path,
            output_path=output_path,
            charset=charset,
            width=width,
            height=height,
            invert=invert,
            brightness=brightness,
            contrast=contrast,
            dithering=dithering,
            color_mode=color_mode,
            **options,
        )

    def get_available_fonts(self) -> List[str]:
        """Return every installed FIGlet font name."""

        return cast(List[str], self._banner_generator.available_fonts())

    def get_available_styles(self) -> Dict[str, Dict[str, Any]]:
        """Return defensive copies of every text style preset."""

        return cast(Dict[str, Dict[str, Any]], get_available_styles())

    def get_available_alphabets(self) -> List[str]:
        """Return every named density, special, and language character set."""

        return cast(List[str], AlphabetManager.list_available_alphabets())

    def save_to_file(self, glyph_art: str, file_path: str | os.PathLike[str]) -> bool:
        """Atomically save UTF-8 text, retaining the historical boolean result."""

        destination = Path(file_path).expanduser()
        try:
            atomic_write_text(destination, glyph_art)
            logger.debug("Saved glyph art to %s", destination)
            return True
        except AtomicWriteError as exc:
            logger.error("Failed to save %s: %s", destination, exc)
            return False

    def preview_font(self, font: str, text: str = "Glyph Forge") -> str:
        """Generate a preview of one FIGlet font."""

        generator = BannerGenerator(font=font, width=self._banner_generator.width)
        return cast(str, generator.generate(text))

    def preview_style(self, style: str, text: str = "Glyph Forge") -> str:
        """Generate a preview of one style preset."""

        return cast(str, self._banner_generator.generate(text, style=style))

    def convert_text_to_art(self, text: str, font: str = "standard") -> str:
        """Render raw FIGlet text without an additional style."""

        generator = BannerGenerator(font=font, width=self._banner_generator.width)
        return cast(str, generator.figlet.renderText(text))


_api_instance: GlyphForgeAPI | None = None
_api_lock = threading.Lock()


def get_api() -> GlyphForgeAPI:
    """Return the process-wide API instance with thread-safe lazy creation."""

    global _api_instance
    if _api_instance is None:
        with _api_lock:
            if _api_instance is None:
                _api_instance = GlyphForgeAPI()
    return _api_instance


__all__ = ["GlyphForgeAPI", "get_api"]

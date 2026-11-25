"""Optional Textual interface for interactive Glyph Forge workflows."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import find_spec
from typing import Any, List, Optional, Tuple

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import (
    Container,
    Horizontal,
    Vertical,
    VerticalScroll,
)
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

from ..api.glyph_api import GlyphForgeAPI, get_api
from ..config.settings import ConfigManager, get_config
from ..exceptions import GlyphForgeInputError
from ..utils.alphabet_manager import AlphabetManager

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎨 Constants and Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GLYPH_FORGE_BANNER = r"""
   ██████╗ ██╗  ██╗   ██╗██████╗ ██╗  ██╗    ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
  ██╔════╝ ██║  ╚██╗ ██╔╝██╔══██╗██║  ██║    ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
  ██║  ███╗██║   ╚████╔╝ ██████╔╝███████║    █████╗  ██║   ██║██████╔╝██║  ███╗█████╗
  ██║   ██║██║    ╚██╔╝  ██╔═══╝ ██╔══██║    ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
  ╚██████╔╝███████╗██║   ██║     ██║  ██║    ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
   ╚═════╝ ╚══════╝╚═╝   ╚═╝     ╚═╝  ╚═╝    ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
"""

COLOR_MODES = [
    ("none", "None (Grayscale)"),
    ("ansi16", "ANSI 16 Colors"),
    ("ansi256", "ANSI 256 Colors"),
    ("truecolor", "Truecolor (24-bit)"),
    ("html", "HTML Output"),
]

DITHER_ALGORITHMS = [
    ("none", "None"),
    ("floyd-steinberg", "Floyd-Steinberg"),
    ("atkinson", "Atkinson"),
]

RESAMPLE_FILTERS = [
    ("nearest", "Nearest"),
    ("bilinear", "Bilinear"),
    ("bicubic", "Bicubic"),
    ("lanczos", "Lanczos (Best)"),
]

BANNER_STYLES = [
    ("minimal", "Minimal"),
    ("boxed", "Boxed"),
    ("shadowed", "Shadowed"),
    ("double", "Double Border"),
    ("metallic", "Metallic"),
    ("circuit", "Circuit"),
    ("eidosian", "Eidosian ⚡"),
]

BANNER_EFFECTS = [
    ("shadow", "Shadow"),
    ("glow", "Glow"),
    ("emboss", "Emboss"),
    ("digital", "Digital"),
    ("fade", "Fade"),
]


def build_stream_command(
    *,
    source_type: str,
    source: str,
    resolution: str,
    fps: str,
    mode: str,
    color: str,
    audio: bool,
    record: bool,
    stats: bool,
) -> list[str]:
    """Build and validate the streaming subprocess command used by the TUI."""

    if source_type not in {"screen", "video", "webcam"}:
        raise GlyphForgeInputError(f"unsupported source type: {source_type}")
    if resolution not in {"auto", "1080p", "720p", "480p", "360p"}:
        raise GlyphForgeInputError(f"unsupported resolution: {resolution}")
    if mode not in {"ascii", "auto", "blocks", "braille", "gradient", "hybrid"}:
        raise GlyphForgeInputError(f"unsupported render mode: {mode}")
    if color not in {"auto", "ansi256", "none", "truecolor"}:
        raise GlyphForgeInputError(f"unsupported color mode: {color}")
    try:
        fps_value = float(fps)
    except ValueError as exc:
        raise GlyphForgeInputError("FPS must be a number") from exc
    if not 1 <= fps_value <= 240:
        raise GlyphForgeInputError("FPS must be between 1 and 240")

    cmd = [sys.executable, "-m", "glyph_forge", "stream"]
    normalized_source = source.strip()
    if source_type == "webcam":
        device = normalized_source or "0"
        if not device.isdigit() or int(device) < 0:
            raise GlyphForgeInputError("webcam device must be a non-negative integer")
        cmd.extend(["--webcam", device])
    elif source_type == "screen":
        cmd.append("--screen")
    else:
        if not normalized_source:
            raise GlyphForgeInputError("a video path or URL is required")
        cmd.append(normalized_source)

    cmd.extend(["--resolution", resolution, "--fps", str(fps_value)])
    cmd.extend(["--mode", mode, "--color", color])
    if not audio:
        cmd.append("--no-audio")
    if record:
        cmd.extend(["--record", "auto"])
    else:
        cmd.extend(["--record", "none"])
    if not stats:
        cmd.append("--no-stats")
    return cmd


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🧩 Custom Widgets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class LabeledInput(Horizontal):
    """Input field with label for consistent form layout."""

    DEFAULT_CSS = """
    LabeledInput {
        height: auto;
        margin: 1 0;
    }
    LabeledInput > Label {
        width: 20;
        padding: 1 1;
    }
    LabeledInput > Input {
        width: 1fr;
    }
    """

    def __init__(
        self,
        label: str,
        input_id: str,
        placeholder: str = "",
        value: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._label_text = label
        self._input_id = input_id
        self._placeholder = placeholder
        self._value = value

    def compose(self) -> ComposeResult:
        yield Label(self._label_text)
        yield Input(
            value=self._value,
            placeholder=self._placeholder,
            id=self._input_id,
        )


class LabeledSelect(Horizontal):
    """Select dropdown with label for consistent form layout."""

    DEFAULT_CSS = """
    LabeledSelect {
        height: auto;
        margin: 1 0;
    }
    LabeledSelect > Label {
        width: 20;
        padding: 1 1;
    }
    LabeledSelect > Select {
        width: 1fr;
    }
    """

    def __init__(
        self,
        label: str,
        options: List[Tuple[str, str]],
        select_id: str,
        value: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._label_text = label
        self._options = options
        self._select_id = select_id
        self._value = value

    def compose(self) -> ComposeResult:
        yield Label(self._label_text)
        yield Select(
            [(name, val) for val, name in self._options],
            id=self._select_id,
            value=self._value or self._options[0][0] if self._options else None,
        )


class LabeledSwitch(Horizontal):
    """Switch toggle with label for consistent form layout."""

    DEFAULT_CSS = """
    LabeledSwitch {
        height: auto;
        margin: 1 0;
    }
    LabeledSwitch > Label {
        width: 20;
        padding: 1 1;
    }
    LabeledSwitch > Switch {
        width: auto;
    }
    """

    def __init__(
        self,
        label: str,
        switch_id: str,
        value: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._label_text = label
        self._switch_id = switch_id
        self._value = value

    def compose(self) -> ComposeResult:
        yield Label(self._label_text)
        yield Switch(value=self._value, id=self._switch_id)


class OutputPanel(VerticalScroll):
    """Scrollable panel for displaying glyph art output."""

    DEFAULT_CSS = """
    OutputPanel {
        border: solid $primary;
        height: 1fr;
        padding: 1;
        background: $surface;
    }
    OutputPanel > Static {
        width: auto;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._content = ""

    def compose(self) -> ComposeResult:
        yield Static("", id="output_content")

    def set_content(self, content: str) -> None:
        """Set the output content."""
        self._content = content
        try:
            self.query_one("#output_content", Static).update(content)
        except Exception:
            pass

    def clear(self) -> None:
        """Clear the output."""
        self.set_content("")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📝 Banner Generator Tab
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BannerTab(Container):
    """Banner generation tab with full parameter control."""

    DEFAULT_CSS = """
    BannerTab {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 2fr;
        padding: 1;
    }
    BannerTab > .controls {
        padding: 1;
        border: solid $primary;
        height: 100%;
    }
    BannerTab > .output {
        padding: 1;
        height: 100%;
    }
    """

    def __init__(self, api: GlyphForgeAPI, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api = api
        self._fonts: List[str] = []

    def compose(self) -> ComposeResult:
        # Load available fonts
        try:
            self._fonts = self.api.get_available_fonts()
        except Exception:
            self._fonts = ["standard", "slant", "big", "banner"]

        # Prioritize common fonts at the top
        priority_fonts = ["slant", "standard", "banner", "big", "doom", "small", "block"]
        available_priority = [f for f in priority_fonts if f in self._fonts]
        other_fonts = [f for f in self._fonts if f not in priority_fonts][:43]  # Keep total ~50
        ordered_fonts = available_priority + other_fonts
        font_options = [(f, f) for f in ordered_fonts]

        default_font = available_priority[0] if available_priority else ordered_fonts[0]

        with Vertical(classes="controls"):
            yield Static("⚡ Banner Generator", classes="section_title")
            yield LabeledInput(
                "Text:",
                "banner_text",
                placeholder="Enter text to transform...",
            )
            yield LabeledSelect(
                "Font:",
                font_options,
                "banner_font",
                value=default_font,
            )
            yield LabeledSelect(
                "Style:",
                BANNER_STYLES,
                "banner_style",
                value="minimal",
            )
            yield LabeledInput(
                "Width:",
                "banner_width",
                placeholder="80",
                value="80",
            )
            yield LabeledSwitch("Color Output:", "banner_color", value=False)

            with Horizontal(classes="button_row"):
                yield Button("⚡ Generate Banner", id="btn_banner_generate", variant="primary")
                yield Button("📋 Preview Fonts", id="btn_banner_preview")
                yield Button("🗑️ Clear", id="btn_banner_clear", variant="warning")

        with Vertical(classes="output"):
            yield Static("Output Preview", classes="section_title")
            yield OutputPanel(id="banner_output")

    @on(Button.Pressed, "#btn_banner_generate")
    def generate_banner(self) -> None:
        """Generate the banner with current settings."""
        text = self.query_one("#banner_text", Input).value
        if not text:
            self._show_output("⚠️ Please enter text to transform")
            return

        try:
            font_select = self.query_one("#banner_font", Select)
            style_select = self.query_one("#banner_style", Select)
            width_input = self.query_one("#banner_width", Input)
            color_switch = self.query_one("#banner_color", Switch)

            font = str(font_select.value) if font_select.value else "slant"
            style = str(style_select.value) if style_select.value else "minimal"
            width = int(width_input.value) if width_input.value else 80
            color = color_switch.value

            banner = self.api.generate_banner(
                text=text,
                font=font,
                style=style,
                width=width,
                color=color,
            )
            self._show_output(banner)
        except Exception as e:
            self._show_output(f"❌ Error: {str(e)}")

    @on(Button.Pressed, "#btn_banner_preview")
    def preview_fonts(self) -> None:
        """Preview available fonts."""
        text = self.query_one("#banner_text", Input).value or "Glyph"
        try:
            preview = self.api._banner_generator.preview_fonts(text, limit=5)
            self._show_output(preview)
        except Exception as e:
            self._show_output(f"❌ Error: {str(e)}")

    @on(Button.Pressed, "#btn_banner_clear")
    def clear_output(self) -> None:
        """Clear the output panel."""
        self._show_output("")

    def _show_output(self, content: str) -> None:
        """Update the output panel."""
        try:
            self.query_one("#banner_output", OutputPanel).set_content(content)
        except Exception:
            pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🖼️ Image Converter Tab
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ImageTab(Container):
    """Image conversion tab with comprehensive parameter control."""

    DEFAULT_CSS = """
    ImageTab {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 2fr;
        padding: 1;
    }
    ImageTab > .controls {
        padding: 1;
        border: solid $primary;
        height: 100%;
        overflow-y: auto;
    }
    ImageTab > .output {
        padding: 1;
        height: 100%;
    }
    """

    def __init__(self, api: GlyphForgeAPI, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api = api
        self._charsets: List[str] = []

    def compose(self) -> ComposeResult:
        # Load available charsets
        try:
            self._charsets = AlphabetManager.list_available_alphabets()
        except Exception:
            self._charsets = ["general", "detailed", "minimal", "blocks"]

        charset_options = [(c, c) for c in self._charsets]

        with VerticalScroll(classes="controls"):
            yield Static("🖼️ Image to Glyph Converter", classes="section_title")

            # File input
            yield LabeledInput(
                "Image Path:",
                "image_path",
                placeholder="/path/to/image.png",
            )

            # Dimensions
            yield Static("━━ Dimensions ━━", classes="subsection")
            yield LabeledInput("Width:", "image_width", placeholder="100", value="100")
            yield LabeledInput("Height:", "image_height", placeholder="auto")
            yield LabeledInput("Aspect Ratio:", "image_aspect", placeholder="0.55", value="0.55")

            # Character set and color
            yield Static("━━ Output Mode ━━", classes="subsection")
            yield LabeledSelect("Charset:", charset_options, "image_charset", value="general")
            yield LabeledSelect("Color Mode:", COLOR_MODES, "image_color", value="none")

            # Dithering
            yield Static("━━ Dithering ━━", classes="subsection")
            yield LabeledSwitch("Enable Dither:", "image_dither", value=False)
            yield LabeledSelect(
                "Algorithm:", DITHER_ALGORITHMS, "image_dither_alg", value="floyd-steinberg"
            )

            # Quality controls
            yield Static("━━ Quality Controls ━━", classes="subsection")
            yield LabeledInput("Brightness:", "image_brightness", placeholder="1.0", value="1.0")
            yield LabeledInput("Contrast:", "image_contrast", placeholder="1.0", value="1.0")
            yield LabeledInput("Gamma:", "image_gamma", placeholder="1.0", value="1.0")
            yield LabeledSelect("Resample:", RESAMPLE_FILTERS, "image_resample", value="lanczos")

            # Filters
            yield Static("━━ Filters ━━", classes="subsection")
            yield LabeledSwitch("Auto Contrast:", "image_autocontrast", value=False)
            yield LabeledSwitch("Equalize:", "image_equalize", value=False)
            yield LabeledSwitch("Invert Image:", "image_invert", value=False)
            yield LabeledSwitch("Edge Enhance:", "image_edge", value=False)
            yield LabeledSwitch("Sharpen:", "image_sharpen", value=False)
            yield LabeledInput("Blur Radius:", "image_blur", placeholder="0.0", value="0.0")
            yield LabeledInput("Posterize Bits:", "image_posterize", placeholder="none")

            with Horizontal(classes="button_row"):
                yield Button("🎨 Convert Image", id="btn_image_convert", variant="primary")
                yield Button("📋 List Charsets", id="btn_image_charsets")
                yield Button("🗑️ Clear", id="btn_image_clear", variant="warning")

        with Vertical(classes="output"):
            yield Static("Output Preview", classes="section_title")
            yield OutputPanel(id="image_output")

    @on(Button.Pressed, "#btn_image_convert")
    def convert_image(self) -> None:
        """Convert image with current settings."""
        image_path = self.query_one("#image_path", Input).value
        if not image_path:
            self._show_output("⚠️ Please enter an image path")
            return

        if not os.path.exists(image_path):
            self._show_output(f"❌ File not found: {image_path}")
            return

        try:
            # Gather all parameters
            width = self._get_int("#image_width", 100)
            height = self._get_int("#image_height", None)
            aspect_ratio = self._get_float("#image_aspect", 0.55)
            charset = self._get_select("#image_charset", "general")
            color_mode = self._get_select("#image_color", "none")
            dithering = self.query_one("#image_dither", Switch).value
            dither_alg = (
                self._get_select("#image_dither_alg", "floyd-steinberg") if dithering else None
            )
            brightness = self._get_float("#image_brightness", 1.0)
            contrast = self._get_float("#image_contrast", 1.0)
            gamma = self._get_float("#image_gamma", 1.0)
            resample = self._get_select("#image_resample", "lanczos")
            autocontrast = self.query_one("#image_autocontrast", Switch).value
            equalize = self.query_one("#image_equalize", Switch).value
            invert_image = self.query_one("#image_invert", Switch).value
            edge_enhance = self.query_one("#image_edge", Switch).value
            sharpen = self.query_one("#image_sharpen", Switch).value
            blur_radius = self._get_float("#image_blur", 0.0)
            posterize_bits = self._get_int("#image_posterize", None)

            result = self.api.image_to_glyph(
                image_path=image_path,
                width=width,
                height=height,
                charset=charset,
                color_mode=color_mode,
                dithering=dithering,
                dither_algorithm=dither_alg,
                brightness=brightness,
                contrast=contrast,
                gamma=gamma,
                aspect_ratio=aspect_ratio,
                resample=resample,
                autocontrast=autocontrast,
                equalize=equalize,
                invert_image=invert_image,
                edge_enhance=edge_enhance,
                sharpen=sharpen,
                blur_radius=blur_radius,
                posterize_bits=posterize_bits,
            )
            self._show_output(result)
        except Exception as e:
            self._show_output(f"❌ Error: {str(e)}")

    @on(Button.Pressed, "#btn_image_charsets")
    def list_charsets(self) -> None:
        """List available character sets."""
        charsets = AlphabetManager.list_available_alphabets()
        output = "Available Character Sets:\n" + "━" * 30 + "\n"
        for cs in charsets:
            try:
                chars = AlphabetManager.get_alphabet(cs)
                preview = chars[:20] + "..." if len(chars) > 20 else chars
                output += f"  {cs}: {preview}\n"
            except Exception:
                output += f"  {cs}\n"
        self._show_output(output)

    @on(Button.Pressed, "#btn_image_clear")
    def clear_output(self) -> None:
        """Clear the output panel."""
        self._show_output("")

    def _get_int(self, selector: str, default: Optional[int]) -> Optional[int]:
        """Get integer value from input."""
        try:
            value = self.query_one(selector, Input).value
            return int(value) if value else default
        except (ValueError, Exception):
            return default

    def _get_float(self, selector: str, default: float) -> float:
        """Get float value from input."""
        try:
            value = self.query_one(selector, Input).value
            return float(value) if value else default
        except (ValueError, Exception):
            return default

    def _get_select(self, selector: str, default: str) -> str:
        """Get select value."""
        try:
            value = self.query_one(selector, Select).value
            return str(value) if value else default
        except Exception:
            return default

    def _show_output(self, content: str) -> None:
        """Update the output panel."""
        try:
            self.query_one("#image_output", OutputPanel).set_content(content)
        except Exception:
            pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📺 Streaming Tab
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class StreamingTab(Container):
    """Streaming controls for video/webcam/YouTube/browser."""

    DEFAULT_CSS = """
    StreamingTab {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 1fr;
        padding: 1;
    }
    StreamingTab > .panel {
        padding: 1;
        border: solid $primary;
        margin: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        # Source selection panel
        with Vertical(classes="panel"):
            yield Static("📺 Stream Source", classes="section_title")

            yield LabeledSelect(
                "Source Type:",
                [
                    ("video", "Video File"),
                    ("youtube", "YouTube URL"),
                    ("webcam", "Webcam"),
                    ("screen", "Screen Capture"),
                ],
                "stream_source_type",
                value="video",
            )

            yield LabeledInput(
                "Source/URL:",
                "stream_source",
                placeholder="Path, URL, or device ID...",
            )

            yield LabeledSelect(
                "Resolution:",
                [
                    ("1080p", "1080p (HD)"),
                    ("720p", "720p (Standard)"),
                    ("480p", "480p (Fast)"),
                    ("auto", "Auto"),
                ],
                "stream_resolution",
                value="720p",
            )

            yield LabeledInput("FPS:", "stream_fps", placeholder="30", value="30")

        # Options panel
        with Vertical(classes="panel"):
            yield Static("⚙️ Stream Options", classes="section_title")

            yield LabeledSelect(
                "Render Mode:",
                [
                    ("auto", "Auto (Terminal Aware)"),
                    ("gradient", "Gradient (Best Quality)"),
                    ("braille", "Braille (High Detail)"),
                    ("ascii", "ASCII (Maximum Compatibility)"),
                    ("blocks", "Unicode Blocks"),
                    ("hybrid", "Hybrid"),
                ],
                "stream_mode",
                value="auto",
            )

            yield LabeledSelect(
                "Color Mode:",
                [
                    ("auto", "Auto (Terminal Aware)"),
                    ("ansi256", "ANSI 256 (Fast)"),
                    ("truecolor", "TrueColor (HQ)"),
                    ("none", "No Color"),
                ],
                "stream_color",
                value="auto",
            )

            yield LabeledSwitch("Audio:", "stream_audio", value=True)
            yield LabeledSwitch("Record:", "stream_record", value=False)
            yield LabeledSwitch("Stats:", "stream_stats", value=True)

            with Horizontal(classes="button_row"):
                yield Button("🚀 Launch Stream", id="btn_stream_start", variant="primary")
                yield Button("⏹️ Stop", id="btn_stream_stop", variant="error")

            yield Static("", id="stream_status")

    @on(Button.Pressed, "#btn_stream_start")
    def start_stream(self) -> None:
        """Launch the streaming process."""
        try:
            source_type = self._get_select("#stream_source_type", "video")
            source = self.query_one("#stream_source", Input).value

            cmd = build_stream_command(
                source_type=source_type,
                source=source,
                resolution=self._get_select("#stream_resolution", "720p"),
                fps=self._get_input("#stream_fps", "30"),
                mode=self._get_select("#stream_mode", "auto"),
                color=self._get_select("#stream_color", "auto"),
                audio=self.query_one("#stream_audio", Switch).value,
                record=self.query_one("#stream_record", Switch).value,
                stats=self.query_one("#stream_stats", Switch).value,
            )

            self._update_status(f"🚀 Launching: {' '.join(cmd)}...")

            # Suspend Textual so the renderer owns the current terminal until
            # playback ends, then restore the application screen.
            with self.app.suspend():
                result = subprocess.run(cmd, check=False)
            if result.returncode == 0:
                self._update_status("✅ Stream ended")
            else:
                self._update_status(f"❌ Stream exited with status {result.returncode}")

        except Exception as e:
            self._update_status(f"❌ Error: {str(e)}")

    @on(Button.Pressed, "#btn_stream_stop")
    def stop_stream(self) -> None:
        """Stop streaming (info message)."""
        self._update_status("ℹ️ Press Ctrl+C while the stream is active to return")

    def _get_input(self, selector: str, default: str) -> str:
        """Get input value."""
        try:
            value = self.query_one(selector, Input).value
            return value if value else default
        except Exception:
            return default

    def _get_select(self, selector: str, default: str) -> str:
        """Get select value."""
        try:
            value = self.query_one(selector, Select).value
            return str(value) if value else default
        except Exception:
            return default

    def _update_status(self, message: str) -> None:
        """Update status display."""
        try:
            self.query_one("#stream_status", Static).update(message)
        except Exception:
            pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⚙️ Settings Tab
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SettingsTab(Container):
    """Settings configuration tab."""

    DEFAULT_CSS = """
    SettingsTab {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 1fr;
        padding: 1;
    }
    SettingsTab > .panel {
        padding: 1;
        border: solid $primary;
        margin: 0 1;
    }
    """

    def __init__(self, config: ConfigManager, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config

    def compose(self) -> ComposeResult:
        # Banner defaults
        with Vertical(classes="panel"):
            yield Static("📝 Banner Defaults", classes="section_title")
            yield LabeledInput(
                "Default Font:",
                "cfg_banner_font",
                value=str(self.config.get("banner", "default_font", "slant")),
            )
            yield LabeledInput(
                "Default Width:",
                "cfg_banner_width",
                value=str(self.config.get("banner", "default_width", 80)),
            )
            yield LabeledInput(
                "Default Style:",
                "cfg_banner_style",
                value=str(self.config.get("banner", "default_style", "minimal")),
            )
            yield LabeledSwitch(
                "Cache Enabled:",
                "cfg_banner_cache",
                value=bool(self.config.get("banner", "cache_enabled", True)),
            )

        # Image defaults
        with Vertical(classes="panel"):
            yield Static("🖼️ Image Defaults", classes="section_title")
            yield LabeledInput(
                "Default Charset:",
                "cfg_image_charset",
                value=str(self.config.get("image", "default_charset", "general")),
            )
            yield LabeledInput(
                "Default Width:",
                "cfg_image_width",
                value=str(self.config.get("image", "default_width", 100)),
            )
            yield LabeledInput(
                "Max Width:",
                "cfg_image_max_width",
                value=str(self.config.get("image", "max_width", 500)),
            )
            yield LabeledInput(
                "Default Gamma:",
                "cfg_image_gamma",
                value=str(self.config.get("image", "gamma", 1.0)),
            )
            yield LabeledInput(
                "Aspect Ratio:",
                "cfg_image_aspect",
                value=str(self.config.get("image", "aspect_ratio", 0.55)),
            )
            yield LabeledSwitch(
                "Dithering:",
                "cfg_image_dither",
                value=bool(self.config.get("image", "dithering", False)),
            )

        # Performance settings
        with Vertical(classes="panel"):
            yield Static("⚡ Performance", classes="section_title")
            yield LabeledInput(
                "Optimization:",
                "cfg_perf_opt",
                value=str(self.config.get("performance", "optimization_level", 3)),
            )
            yield LabeledSwitch(
                "Cache Enabled:",
                "cfg_perf_cache",
                value=bool(self.config.get("performance", "cache_enabled", True)),
            )
            yield LabeledSwitch(
                "Lazy Loading:",
                "cfg_perf_lazy",
                value=bool(self.config.get("performance", "lazy_loading", True)),
            )
            yield LabeledSwitch(
                "Debug Mode:",
                "cfg_perf_debug",
                value=bool(self.config.get("performance", "debug_mode", False)),
            )

        # IO settings
        with Vertical(classes="panel"):
            yield Static("💾 Output Settings", classes="section_title")
            yield LabeledSelect(
                "Output Format:",
                [("text", "Text"), ("html", "HTML"), ("svg", "SVG")],
                "cfg_io_format",
                value=str(self.config.get("io", "output_format", "text")),
            )
            yield LabeledSwitch(
                "Color Output:",
                "cfg_io_color",
                value=bool(self.config.get("io", "color_output", True)),
            )
            yield LabeledSwitch(
                "Auto Terminal:",
                "cfg_io_terminal",
                value=bool(self.config.get("io", "auto_detect_terminal", True)),
            )

            with Horizontal(classes="button_row"):
                yield Button("Save Settings", id="btn_settings_save", variant="primary")
                yield Button("Reset Defaults", id="btn_settings_reset", variant="warning")

            yield Static("", id="settings_status")

    @on(Button.Pressed, "#btn_settings_save")
    def save_settings(self) -> None:
        """Save current settings."""
        try:
            # Banner settings
            self.config.set("banner", "default_font", self._get_input("#cfg_banner_font"))
            self.config.set(
                "banner", "default_width", int(self._get_input("#cfg_banner_width", "80"))
            )
            self.config.set("banner", "default_style", self._get_input("#cfg_banner_style"))
            self.config.set("banner", "cache_enabled", self._get_switch("#cfg_banner_cache"))

            # Image settings
            self.config.set("image", "default_charset", self._get_input("#cfg_image_charset"))
            self.config.set(
                "image", "default_width", int(self._get_input("#cfg_image_width", "100"))
            )
            self.config.set(
                "image", "max_width", int(self._get_input("#cfg_image_max_width", "500"))
            )
            self.config.set("image", "gamma", float(self._get_input("#cfg_image_gamma", "1.0")))
            self.config.set(
                "image", "aspect_ratio", float(self._get_input("#cfg_image_aspect", "0.55"))
            )
            self.config.set("image", "dithering", self._get_switch("#cfg_image_dither"))

            # Performance settings
            self.config.set(
                "performance", "optimization_level", int(self._get_input("#cfg_perf_opt", "3"))
            )
            self.config.set("performance", "cache_enabled", self._get_switch("#cfg_perf_cache"))
            self.config.set("performance", "lazy_loading", self._get_switch("#cfg_perf_lazy"))
            self.config.set("performance", "debug_mode", self._get_switch("#cfg_perf_debug"))

            # IO settings
            self.config.set("io", "color_output", self._get_switch("#cfg_io_color"))
            self.config.set("io", "auto_detect_terminal", self._get_switch("#cfg_io_terminal"))

            self._update_status("✅ Settings saved successfully")
        except Exception as e:
            self._update_status(f"❌ Error saving: {str(e)}")

    @on(Button.Pressed, "#btn_settings_reset")
    def reset_settings(self) -> None:
        """Reset to default settings."""
        self._update_status("ℹ️ Restart application to apply defaults")

    def _get_input(self, selector: str, default: str = "") -> str:
        """Get input value."""
        try:
            return self.query_one(selector, Input).value or default
        except Exception:
            return default

    def _get_switch(self, selector: str) -> bool:
        """Get switch value."""
        try:
            return self.query_one(selector, Switch).value
        except Exception:
            return False

    def _update_status(self, message: str) -> None:
        """Update status display."""
        try:
            self.query_one("#settings_status", Static).update(message)
        except Exception:
            pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ℹ️ About Tab
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AboutTab(Container):
    """About/Help tab with documentation and info."""

    DEFAULT_CSS = """
    AboutTab {
        padding: 2;
    }
    AboutTab > .banner {
        text-align: center;
        color: $primary;
    }
    AboutTab > .info {
        margin: 2 0;
        padding: 1;
        border: solid $primary;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(GLYPH_FORGE_BANNER, classes="banner")
        yield Static(
            "Adaptive glyph rendering for terminals",
            classes="tagline",
        )

        with Vertical(classes="info"):
            yield Static("📖 Quick Reference", classes="section_title")
            yield Static(
                """
[bold]Banner Generator[/bold]
  Transform text into stylized ASCII/Unicode art banners.
  Choose from 100+ fonts and multiple styling presets.

[bold]Image Converter[/bold]
  Convert images to high-fidelity glyph art.
  Supports grayscale, ANSI colors, and HTML output.
  Advanced options: dithering, gamma, edge enhancement.

[bold]Streaming[/bold]
  Real-time glyph rendering for video, webcam, and YouTube.
  Launch browser sessions rendered as live glyph art.

[bold]Keyboard Shortcuts[/bold]
  q / Ctrl+C  - Quit application
  Tab         - Navigate between fields
  Enter       - Activate buttons
  Escape      - Close dialogs
""",
                markup=True,
            )

        with Vertical(classes="info"):
            yield Static("🔧 Dependencies", classes="section_title")
            yield Static(self._get_dependency_status())

    def _get_dependency_status(self) -> str:
        """Check and report dependency status."""
        deps = []

        # Core dependencies
        deps.append("✅ Pillow (Image processing)")
        deps.append("✅ NumPy (Array operations)")
        deps.append("✅ Rich (Terminal formatting)")
        deps.append("✅ Textual (TUI framework)")

        # Optional dependencies
        if find_spec("cv2") is not None:
            deps.append("✅ OpenCV (Video processing)")
        else:
            deps.append("⚠️ OpenCV (pip install opencv-python)")

        if find_spec("yt_dlp") is not None:
            deps.append("✅ yt-dlp (YouTube streaming)")
        else:
            deps.append("⚠️ yt-dlp (pip install yt-dlp)")

        if find_spec("mss") is not None:
            deps.append("✅ mss (Screen capture)")
        else:
            deps.append("⚠️ mss (pip install mss)")

        return "\n".join(deps)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🚀 Main Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class GlyphForgeApp(App[None]):
    """Interactive banner, image, streaming, and settings application."""

    TITLE = "⚡ Glyph Forge ⚡"
    SUB_TITLE = "Adaptive terminal rendering"
    CSS_PATH = "glyph_forge.css"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("f1", "show_help", "Help", show=True),
        Binding("ctrl+s", "save", "Save", show=True),
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api: Optional[GlyphForgeAPI] = None
        self.config: Optional[ConfigManager] = None

    def compose(self) -> ComposeResult:
        """Compose the main application interface."""
        yield Header()

        with TabbedContent(initial="banner"):
            with TabPane("📝 Banner", id="banner"):
                yield BannerTab(self._get_api())
            with TabPane("🖼️ Image", id="image"):
                yield ImageTab(self._get_api())
            with TabPane("📺 Streaming", id="streaming"):
                yield StreamingTab()
            with TabPane("⚙️ Settings", id="settings"):
                yield SettingsTab(self._get_config())
            with TabPane("ℹ️ About", id="about"):
                yield AboutTab()

        yield Footer()

    def on_mount(self) -> None:
        """Initialize on mount."""
        self.title = "⚡ Glyph Forge ⚡"
        self.sub_title = "Adaptive terminal rendering"

    def _get_api(self) -> GlyphForgeAPI:
        """Get or create API instance."""
        if self.api is None:
            self.api = get_api()
        return self.api

    def _get_config(self) -> ConfigManager:
        """Get or create config instance."""
        if self.config is None:
            self.config = get_config()
        return self.config

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_show_help(self) -> None:
        """Show help (switch to About tab)."""
        try:
            tabs = self.query_one(TabbedContent)
            tabs.active = "about"
        except Exception:
            pass

    def action_save(self) -> None:
        """Trigger save in settings tab."""
        try:
            settings_tab = self.query_one(SettingsTab)
            settings_tab.save_settings()
        except Exception:
            pass


def run_tui() -> None:
    """Entry point for running the TUI."""
    app = GlyphForgeApp()
    app.run()


if __name__ == "__main__":
    run_tui()

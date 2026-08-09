"""Full-screen Textual interface for Glyph Forge."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterable

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DirectoryTree,
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
from textual.worker import get_current_worker

from ..runtime import runtime_report
from ..utils.alphabet_manager import AlphabetManager

MEDIA_EXTENSIONS = {
    ".apng",
    ".avi",
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".png",
    ".tif",
    ".tiff",
    ".webm",
    ".webp",
}


def filter_media_paths(paths: Iterable[Path]) -> list[Path]:
    """Keep visible folders and media files for the TUI browser."""

    return [
        path
        for path in paths
        if not path.name.startswith(".")
        and (path.is_dir() or path.suffix.casefold() in MEDIA_EXTENSIONS)
    ]


class MediaDirectoryTree(DirectoryTree):
    """Filesystem browser filtered to folders and supported visual media."""

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return filter_media_paths(paths)


class FilePicker(ModalScreen[Path | None]):
    """Keyboard- and mouse-friendly media picker."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, root: Path | None = None) -> None:
        super().__init__()
        self.root = root or Path.home()

    def compose(self) -> ComposeResult:
        with Vertical(id="file-picker-dialog"):
            yield Static("Choose an image or video", classes="dialog-title")
            yield MediaDirectoryTree(self.root, id="media-tree")
            with Horizontal(classes="dialog-actions"):
                yield Button("Cancel", id="picker-cancel", variant="default")

    @on(DirectoryTree.FileSelected)
    def select_file(self, event: DirectoryTree.FileSelected) -> None:
        self.dismiss(event.path)

    @on(Button.Pressed, "#picker-cancel")
    def cancel_button(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class GlyphForgeApp(App[str | None]):
    """Responsive TUI for still, text, and live glyph workflows."""

    TITLE = "Glyph Forge"
    SUB_TITLE = "Portable visual foundry"
    CSS_PATH = "glyph_forge.css"
    ENABLE_COMMAND_PALETTE = True
    BINDINGS = [
        ("ctrl+o", "browse", "Open media"),
        ("ctrl+s", "save", "Save result"),
        ("ctrl+g", "studio", "Browser studio"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.image_result = ""
        self.text_result = ""
        self._live_stop = threading.Event()

    def compose(self) -> ComposeResult:
        charsets = [
            (name.replace("_", " ").title(), name)
            for name in sorted(AlphabetManager.list_available_alphabets())
        ]
        yield Header(show_clock=True)
        with TabbedContent(initial="image-tab", id="workspace-tabs"):
            with TabPane("Image", id="image-tab"):
                with Horizontal(classes="workflow"):
                    with Vertical(classes="form-panel"):
                        yield Static("IMAGE → GLYPHS", classes="eyebrow")
                        yield Label("Source")
                        with Horizontal(classes="input-row"):
                            yield Input(
                                placeholder="Path to an image",
                                id="image-path",
                            )
                            yield Button("Browse", id="image-browse")
                        yield Label("Character set")
                        yield Select(
                            charsets,
                            value="general",
                            allow_blank=False,
                            id="image-charset",
                        )
                        with Horizontal(classes="field-row"):
                            with Vertical():
                                yield Label("Width")
                                yield Input("80", type="integer", id="image-width")
                            with Vertical():
                                yield Label("Colour")
                                yield Select(
                                    [
                                        ("Plain", "none"),
                                        ("ANSI true colour", "ansi"),
                                        ("HTML", "html"),
                                    ],
                                    value="none",
                                    allow_blank=False,
                                    id="image-color",
                                )
                        with Horizontal(classes="switch-row"):
                            yield Label("Invert")
                            yield Switch(id="image-invert")
                            yield Label("Dither")
                            yield Switch(id="image-dither")
                        yield Button(
                            "Forge preview",
                            id="image-convert",
                            variant="success",
                            classes="primary-action",
                        )
                        yield Label("Save as")
                        with Horizontal(classes="input-row"):
                            yield Input(
                                placeholder="output.txt",
                                id="image-output-path",
                            )
                            yield Button("Save", id="image-save")
                    with Vertical(classes="preview-panel"):
                        yield Static("PREVIEW", classes="eyebrow")
                        with VerticalScroll(classes="preview-scroll"):
                            yield Static(
                                "Choose an image to begin.",
                                id="image-preview",
                                markup=False,
                                classes="glyph-preview",
                            )

            with TabPane("Text", id="text-tab"):
                with Horizontal(classes="workflow"):
                    with Vertical(classes="form-panel"):
                        yield Static("TEXT → BANNER", classes="eyebrow")
                        yield Label("Words")
                        yield Input("Glyph Forge", id="text-value")
                        with Horizontal(classes="field-row"):
                            with Vertical():
                                yield Label("FIGlet font")
                                yield Input("slant", id="text-font")
                            with Vertical():
                                yield Label("Style")
                                yield Input("minimal", id="text-style")
                        yield Label("Width")
                        yield Input("80", type="integer", id="text-width")
                        yield Button(
                            "Forge banner",
                            id="text-convert",
                            variant="success",
                            classes="primary-action",
                        )
                        yield Label("Save as")
                        with Horizontal(classes="input-row"):
                            yield Input(
                                placeholder="banner.txt",
                                id="text-output-path",
                            )
                            yield Button("Save", id="text-save")
                    with Vertical(classes="preview-panel"):
                        yield Static("PREVIEW", classes="eyebrow")
                        with VerticalScroll(classes="preview-scroll"):
                            yield Static(
                                "Create a banner to begin.",
                                id="text-preview",
                                markup=False,
                                classes="glyph-preview",
                            )

            with TabPane("Live", id="live-tab"):
                with Horizontal(classes="workflow"):
                    with Vertical(classes="form-panel"):
                        yield Static("LIVE → LATEST FRAME", classes="eyebrow")
                        yield Label("Source")
                        with Horizontal(classes="input-row"):
                            yield Input("camera:0", id="live-source")
                            yield Button("Browse", id="live-browse")
                        yield Static(
                            "Use camera:0, screen:1, or a video path.",
                            classes="hint",
                        )
                        yield Label("Fidelity mode")
                        yield Select(
                            [
                                ("Braille · 2×4 subpixels", "braille"),
                                ("Density glyphs", "glyph"),
                                ("True-colour half blocks", "half-block"),
                                ("Quadrants · 2×2 subpixels", "quadrant"),
                            ],
                            value="braille",
                            allow_blank=False,
                            id="live-mode",
                        )
                        with Horizontal(classes="field-row"):
                            with Vertical():
                                yield Label("Columns")
                                yield Input("80", type="integer", id="live-width")
                            with Vertical():
                                yield Label("Target FPS")
                                yield Input("20", type="number", id="live-fps")
                        with Horizontal(classes="action-row"):
                            yield Button(
                                "Start",
                                id="live-start",
                                variant="success",
                                classes="primary-action",
                            )
                            yield Button("Stop", id="live-stop", disabled=True)
                    with Vertical(classes="preview-panel"):
                        with Horizontal(classes="preview-heading"):
                            yield Static("LIVE PREVIEW", classes="eyebrow")
                            yield Static("Idle", id="live-metrics", classes="metrics")
                        with VerticalScroll(classes="preview-scroll"):
                            yield Static(
                                "Start a source to render its newest frame.",
                                id="live-preview",
                                markup=False,
                                classes="glyph-preview",
                            )

            with TabPane("Runtime", id="runtime-tab"):
                with Vertical(classes="runtime-panel"):
                    with Horizontal(classes="runtime-heading"):
                        with Vertical():
                            yield Static("RUNTIME DOCTOR", classes="eyebrow")
                            yield Static(
                                "Hardware profile and optional feature readiness.",
                                classes="hint",
                            )
                        yield Button("Refresh", id="runtime-refresh")
                        yield Button(
                            "Open browser studio",
                            id="runtime-studio",
                            variant="success",
                        )
                    yield Static(id="runtime-report")
        yield Footer()

    def on_mount(self) -> None:
        self.update_runtime_report()

    def selected_value(self, widget_id: str, fallback: str) -> str:
        value = self.query_one(widget_id, Select).value
        return value if isinstance(value, str) else fallback

    def positive_int(self, widget_id: str, fallback: int) -> int:
        value = self.query_one(widget_id, Input).value.strip()
        try:
            return max(1, int(value))
        except ValueError:
            return fallback

    def positive_float(self, widget_id: str, fallback: float) -> float:
        value = self.query_one(widget_id, Input).value.strip()
        try:
            return max(0.1, float(value))
        except ValueError:
            return fallback

    def choose_media(self, destination: str) -> None:
        def selected(path: Path | None) -> None:
            if path is not None:
                self.query_one(destination, Input).value = str(path)

        self.push_screen(FilePicker(), selected)

    @on(Button.Pressed, "#image-browse")
    def browse_image(self) -> None:
        self.choose_media("#image-path")

    @on(Button.Pressed, "#live-browse")
    def browse_live(self) -> None:
        self.choose_media("#live-source")

    @on(Button.Pressed, "#image-convert")
    def request_image_conversion(self) -> None:
        path = self.query_one("#image-path", Input).value.strip()
        if not path:
            self.notify("Choose an image first", severity="warning")
            return
        self.query_one("#image-preview", Static).update("Forging…")
        self.convert_image(
            path,
            self.positive_int("#image-width", 80),
            self.selected_value("#image-charset", "general"),
            self.selected_value("#image-color", "none"),
            self.query_one("#image-invert", Switch).value,
            self.query_one("#image-dither", Switch).value,
        )

    @work(thread=True, exclusive=True, group="image-preview")
    def convert_image(
        self,
        path: str,
        width: int,
        charset: str,
        color: str,
        invert: bool,
        dither: bool,
    ) -> None:
        from ..services.image_to_glyph import ImageGlyphConverter

        converter = ImageGlyphConverter(
            width=width,
            charset=charset,
            invert=invert,
            dithering=dither,
            auto_scale=False,
        )
        if color == "none":
            result = converter.convert(path)
            renderable: str | Text = result
        else:
            result = converter.convert_color(path, color_mode=color)
            renderable = Text.from_ansi(result) if color == "ansi" else result
        worker = get_current_worker()
        if not worker.is_cancelled:
            self.image_result = result
            self.app.call_from_thread(
                self.query_one("#image-preview", Static).update,
                renderable,
            )
            self.app.call_from_thread(self.notify, "Image preview ready")

    @on(Button.Pressed, "#text-convert")
    def request_text_conversion(self) -> None:
        self.convert_text(
            self.query_one("#text-value", Input).value,
            self.query_one("#text-font", Input).value or "slant",
            self.query_one("#text-style", Input).value or "minimal",
            self.positive_int("#text-width", 80),
        )

    @work(thread=True, exclusive=True, group="text-preview")
    def convert_text(self, text: str, font: str, style: str, width: int) -> None:
        from ..services.text_to_banner import text_to_banner

        result = text_to_banner(text, font=font, style=style, width=width)
        worker = get_current_worker()
        if not worker.is_cancelled:
            self.text_result = result
            self.app.call_from_thread(
                self.query_one("#text-preview", Static).update,
                result,
            )
            self.app.call_from_thread(self.notify, "Banner ready")

    def save_result(self, result: str, destination: str) -> None:
        if not result:
            self.notify("Create a result before saving", severity="warning")
            return
        path_value = self.query_one(destination, Input).value.strip()
        if not path_value:
            self.notify("Choose an output path", severity="warning")
            return
        try:
            path = Path(path_value).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(result, encoding="utf-8")
        except OSError as exc:
            self.notify(f"Could not save: {exc}", severity="error")
            return
        self.notify(f"Saved {path}")

    @on(Button.Pressed, "#image-save")
    def save_image(self) -> None:
        self.save_result(self.image_result, "#image-output-path")

    @on(Button.Pressed, "#text-save")
    def save_text(self) -> None:
        self.save_result(self.text_result, "#text-output-path")

    @on(Button.Pressed, "#live-start")
    def start_live(self) -> None:
        self._live_stop.set()
        self._live_stop = threading.Event()
        self.query_one("#live-start", Button).disabled = True
        self.query_one("#live-stop", Button).disabled = False
        self.query_one("#live-preview", Static).update("Opening source…")
        self.run_live(
            self.query_one("#live-source", Input).value.strip() or "camera:0",
            self.selected_value("#live-mode", "braille"),
            self.positive_int("#live-width", 80),
            self.positive_float("#live-fps", 20),
            self._live_stop,
        )

    @on(Button.Pressed, "#live-stop")
    def stop_live(self) -> None:
        self._live_stop.set()
        self.query_one("#live-stop", Button).disabled = True

    @work(thread=True, exclusive=True, group="live-preview")
    def run_live(
        self,
        specification: str,
        mode: str,
        width: int,
        fps: float,
        stop_event: threading.Event,
    ) -> None:
        from ..live.capture import CaptureError, LatestFramePump, create_frame_source
        from ..live.renderers import FrameRenderer, RenderConfig

        color = "truecolor" if mode == "half-block" else "ansi256"
        renderer = FrameRenderer(
            RenderConfig(width=width, mode=mode, color=color, charset="detailed")
        )
        pump: LatestFramePump | None = None
        sequence = 0
        displayed = 0
        dropped = 0
        try:
            source = create_frame_source(specification, fps=fps, loop=True)
            pump = LatestFramePump(source).start()
            worker = get_current_worker()
            while not stop_event.is_set() and not worker.is_cancelled:
                frame = pump.next_frame(sequence, timeout=1 / fps)
                if frame is None:
                    if pump.ended:
                        break
                    continue
                dropped += max(0, frame.sequence - sequence - 1)
                sequence = frame.sequence
                displayed += 1
                result = renderer.render(frame.pixels)
                preview = Text.from_ansi(result.text)
                self.app.call_from_thread(
                    self.query_one("#live-preview", Static).update,
                    preview,
                    layout=False,
                )
                self.app.call_from_thread(
                    self.query_one("#live-metrics", Static).update,
                    f"{displayed} shown · {dropped} dropped",
                )
        except (CaptureError, ValueError) as exc:
            self.app.call_from_thread(
                self.query_one("#live-preview", Static).update,
                f"Could not start live view:\n{exc}",
            )
            self.app.call_from_thread(self.notify, str(exc), severity="error")
        finally:
            if pump is not None:
                pump.stop()
            self.app.call_from_thread(self._live_finished)

    def _live_finished(self) -> None:
        self.query_one("#live-start", Button).disabled = False
        self.query_one("#live-stop", Button).disabled = True

    @on(Button.Pressed, "#runtime-refresh")
    def refresh_runtime(self) -> None:
        self.update_runtime_report()

    def update_runtime_report(self) -> None:
        report = runtime_report()
        profile = report["profile"]
        table = Table(expand=True, show_lines=False)
        table.add_column("Status", width=10)
        table.add_column("Capability", style="bold")
        table.add_column("Purpose")
        table.add_column("Action")
        for item in report["capabilities"]:
            table.add_row(
                (
                    "[green]ready[/green]"
                    if item["available"]
                    else "[yellow]optional[/yellow]"
                ),
                item["label"],
                item["purpose"],
                "" if item["available"] else (item["install_hint"] or ""),
            )
        summary = Text.from_markup(
            f"[bold cyan]{profile['tier']}[/bold cyan] profile · "
            f"{profile['cpu_count']} CPUs · {profile['workers']} workers · "
            f"{profile['stream_width']} live columns · {profile['target_fps']} FPS\n\n"
        )
        self.query_one("#runtime-report", Static).update(Group(summary, table))

    @on(Button.Pressed, "#runtime-studio")
    def open_studio(self) -> None:
        self.action_studio()

    def action_browse(self) -> None:
        active = self.query_one("#workspace-tabs", TabbedContent).active
        self.choose_media("#live-source" if active == "live-tab" else "#image-path")

    def action_save(self) -> None:
        active = self.query_one("#workspace-tabs", TabbedContent).active
        if active == "text-tab":
            self.save_text()
        else:
            self.save_image()

    def action_studio(self) -> None:
        self._live_stop.set()
        self.exit("studio")

    def action_quit(self) -> None:
        self._live_stop.set()
        self.exit(None)

    def on_unmount(self) -> None:
        self._live_stop.set()


__all__ = [
    "FilePicker",
    "GlyphForgeApp",
    "MediaDirectoryTree",
    "filter_media_paths",
]

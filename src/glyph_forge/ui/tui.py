"""Full-screen Textual interface for Glyph Forge."""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

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

from ..persistence import AtomicWriteError, atomic_write_text
from ..runtime import runtime_report
from ..utils.alphabet_manager import AlphabetManager

if TYPE_CHECKING:
    from ..batch import BatchProgress, CancellationToken
    from ..contracts import RenderArtifact, RenderRequest
    from ..projects import ProjectSession

_PIXEL_SIZE = re.compile(r"^\s*(\d+)\s*[x×]\s*(\d+)\s*$", re.IGNORECASE)

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
DOCUMENT_SUFFIXES = (".glyphforge.json", ".glyphpreset.json")


def filter_media_paths(paths: Iterable[Path]) -> list[Path]:
    """Keep visible folders and media files for the TUI browser."""

    return [
        path
        for path in paths
        if not path.name.startswith(".")
        and (path.is_dir() or path.suffix.casefold() in MEDIA_EXTENSIONS)
    ]


def parse_pixel_dimensions(value: str) -> tuple[int, int]:
    """Parse and bound an exact graphical export size from the TUI."""

    match = _PIXEL_SIZE.fullmatch(value)
    if match is None:
        raise ValueError("Output pixels must use WIDTHxHEIGHT, such as 1920x1080")
    width, height = (int(part) for part in match.groups())
    if not 1 <= width <= 8192 or not 1 <= height <= 8192:
        raise ValueError("Output pixels must be between 1 and 8192 per dimension")
    return width, height


class MediaDirectoryTree(DirectoryTree):
    """Filesystem browser filtered to folders and supported visual media."""

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return filter_media_paths(paths)


class DocumentDirectoryTree(DirectoryTree):
    """Filesystem browser limited to project and preset documents."""

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [
            path
            for path in paths
            if not path.name.startswith(".")
            and (
                path.is_dir()
                or any(path.name.endswith(suffix) for suffix in DOCUMENT_SUFFIXES)
            )
        ]


class FilePicker(ModalScreen[Path | None]):
    """Keyboard- and mouse-friendly media or project-document picker."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(
        self,
        root: Path | None = None,
        *,
        documents: bool = False,
    ) -> None:
        super().__init__()
        self.root = root or Path.home()
        self.documents = documents

    def compose(self) -> ComposeResult:
        with Vertical(id="file-picker-dialog"):
            yield Static(
                "Choose a project or preset"
                if self.documents
                else "Choose an image or video",
                classes="dialog-title",
            )
            tree_type = DocumentDirectoryTree if self.documents else MediaDirectoryTree
            yield tree_type(self.root, id="media-tree")
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
        ("ctrl+z", "undo", "Undo project edit"),
        ("ctrl+y", "redo", "Redo project edit"),
        ("ctrl+g", "studio", "Browser studio"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.image_result = ""
        self.text_result = ""
        self._image_source: str | None = None
        self._image_request: RenderRequest | None = None
        self._image_artifact: RenderArtifact | None = None
        self._project_session: ProjectSession | None = None
        self._syncing_project = False
        self._batch_sources: list[Path] = []
        self._batch_cancellation: CancellationToken | None = None
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
                    with VerticalScroll(classes="form-panel"):
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
                        yield Label("Fidelity mode")
                        yield Select(
                            [
                                ("Density glyphs", "glyph"),
                                ("Directional edges", "edge"),
                                ("Braille · 2×4 subpixels", "braille"),
                                ("True-colour half blocks", "half-block"),
                                ("Quadrants · 2×2 subpixels", "quadrant"),
                            ],
                            value="glyph",
                            allow_blank=False,
                            id="image-mode",
                        )
                        with Horizontal(classes="field-row"):
                            with Vertical():
                                yield Label("Columns")
                                yield Input("80", type="integer", id="image-width")
                            with Vertical():
                                yield Label("Rows")
                                yield Input(
                                    "",
                                    placeholder="Auto",
                                    type="integer",
                                    id="image-height",
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
                        with Horizontal(classes="field-row"):
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
                            with Vertical():
                                yield Label("Output pixels")
                                yield Input("1280x720", id="image-output-size")
                        with Horizontal(classes="field-row"):
                            with Vertical():
                                yield Label("Brightness")
                                yield Input(
                                    "1.12", type="number", id="image-brightness"
                                )
                            with Vertical():
                                yield Label("Contrast")
                                yield Input("1.08", type="number", id="image-contrast")
                        yield Label("Graphical export fit")
                        yield Select(
                            [
                                ("Contain · preserve all", "contain"),
                                ("Cover · fill and crop", "cover"),
                                ("Stretch · exact edges", "stretch"),
                            ],
                            value="contain",
                            allow_blank=False,
                            id="image-fit",
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

            with TabPane("Project", id="project-tab"):
                with Horizontal(classes="workflow"):
                    with VerticalScroll(classes="form-panel"):
                        yield Static("PROJECT → SESSION", classes="eyebrow")
                        yield Label("Project document")
                        with Horizontal(classes="input-row"):
                            yield Input(
                                placeholder="artwork.glyphforge.json",
                                id="project-path",
                            )
                            yield Button("Browse", id="project-browse")
                        with Horizontal(classes="action-row compact-actions"):
                            yield Button("Open", id="project-open")
                            yield Button("New", id="project-new")
                            yield Button(
                                "Save",
                                id="project-save",
                                variant="success",
                                disabled=True,
                            )
                        yield Static(
                            "Open a project or create one from the current image.",
                            id="project-status",
                            classes="hint status-block",
                        )
                        yield Label("Recent projects")
                        with Horizontal(classes="input-row"):
                            yield Select(
                                [("None yet", "")],
                                value="",
                                allow_blank=False,
                                id="project-recent",
                                disabled=True,
                            )
                            yield Button(
                                "Open",
                                id="project-recent-open",
                                disabled=True,
                            )
                        yield Label("Active variant")
                        yield Select(
                            [("Default", "default")],
                            value="default",
                            allow_blank=False,
                            id="project-variant",
                            disabled=True,
                        )
                        yield Input(
                            placeholder="New variant name",
                            id="project-variant-name",
                        )
                        with Horizontal(classes="action-row compact-actions"):
                            yield Button("Add", id="project-variant-add", disabled=True)
                            yield Button(
                                "Remove", id="project-variant-remove", disabled=True
                            )
                        with Horizontal(classes="action-row compact-actions"):
                            yield Button("Undo", id="project-undo", disabled=True)
                            yield Button("Redo", id="project-redo", disabled=True)
                        yield Label("Preset document")
                        with Horizontal(classes="input-row"):
                            yield Input(
                                placeholder="look.glyphpreset.json",
                                id="preset-path",
                            )
                            yield Button("Browse", id="preset-browse")
                        with Horizontal(classes="action-row compact-actions"):
                            yield Button("Apply", id="preset-apply")
                            yield Button("Export", id="preset-export")
                    with VerticalScroll(classes="project-panel"):
                        yield Static("SESSION", classes="eyebrow")
                        yield Static(
                            "No project open.",
                            id="project-summary",
                            classes="project-summary",
                        )
                        yield Static("BATCH QUEUE", classes="eyebrow section-heading")
                        yield Static(
                            "No queued images.",
                            id="batch-queue",
                            classes="queue-list",
                        )
                        with Horizontal(classes="action-row queue-actions"):
                            yield Button("Add current image", id="batch-add")
                            yield Button("Clear", id="batch-clear", disabled=True)
                        with Horizontal(classes="field-row"):
                            with Vertical():
                                yield Label("Output folder")
                                yield Input("glyph-forge-output", id="batch-output")
                            with Vertical():
                                yield Label("Workers")
                                yield Input("1", type="integer", id="batch-workers")
                        with Horizontal(classes="action-row queue-actions"):
                            yield Button(
                                "Run batch",
                                id="batch-run",
                                variant="success",
                                disabled=True,
                            )
                            yield Button("Cancel", id="batch-cancel", disabled=True)
                        yield Static(
                            "Queues are bounded and outputs are saved atomically.",
                            id="batch-status",
                            classes="hint status-block",
                        )

            with TabPane("Text", id="text-tab"):
                with Horizontal(classes="workflow"):
                    with VerticalScroll(classes="form-panel"):
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
                    with VerticalScroll(classes="form-panel"):
                        yield Static("LIVE → LATEST FRAME", classes="eyebrow")
                        yield Label("Source")
                        with Horizontal(classes="input-row"):
                            yield Input("camera:0", id="live-source")
                            yield Button("Browse", id="live-browse")
                        yield Static(
                            "Use camera:0, screen:1, a video path, or an https URL.",
                            classes="hint",
                        )
                        yield Label("Fidelity mode")
                        yield Select(
                            [
                                ("Braille · 2×4 subpixels", "braille"),
                                ("Density glyphs", "glyph"),
                                ("Directional edges", "edge"),
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
        self._refresh_project_status()
        self._refresh_recent_projects()
        self._refresh_batch_queue()

    def selected_value(self, widget_id: str, fallback: str) -> str:
        value = self.query_one(widget_id, Select).value
        return value if isinstance(value, str) else fallback

    def positive_int(self, widget_id: str, fallback: int) -> int:
        value = self.query_one(widget_id, Input).value.strip()
        try:
            return max(1, int(value))
        except ValueError:
            return fallback

    def optional_positive_int(self, widget_id: str) -> int | None:
        value = self.query_one(widget_id, Input).value.strip()
        if not value:
            return None
        try:
            return max(1, int(value))
        except ValueError:
            return None

    def bounded_float(
        self,
        widget_id: str,
        fallback: float,
        minimum: float = 0.0,
        maximum: float = 2.0,
    ) -> float:
        value = self.query_one(widget_id, Input).value.strip()
        try:
            return max(minimum, min(maximum, float(value)))
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

    def choose_document(self, destination: str) -> None:
        def selected(path: Path | None) -> None:
            if path is not None:
                self.query_one(destination, Input).value = str(path)

        self.push_screen(FilePicker(documents=True), selected)

    @on(Button.Pressed, "#project-browse")
    def browse_project(self) -> None:
        self.choose_document("#project-path")

    @on(Button.Pressed, "#preset-browse")
    def browse_preset(self) -> None:
        self.choose_document("#preset-path")

    def _project_path(self) -> Path | None:
        value = self.query_one("#project-path", Input).value.strip()
        return Path(value).expanduser() if value else None

    def _apply_request_to_controls(self, request: RenderRequest) -> None:
        charset = self.query_one("#image-charset", Select)
        charset_options = [
            (name.replace("_", " ").title(), name)
            for name in sorted(AlphabetManager.list_available_alphabets())
        ]
        if request.charset not in {value for _, value in charset_options}:
            charset_options.append(
                (f"Custom · {request.charset[:20]}", request.charset)
            )
        charset.set_options(charset_options)
        charset.value = request.charset

        mode = self.query_one("#image-mode", Select)
        mode_options = [
            ("Density glyphs", "glyph"),
            ("Directional edges", "edge"),
            ("Braille · 2×4 subpixels", "braille"),
            ("True-colour half blocks", "half-block"),
            ("Quadrants · 2×2 subpixels", "quadrant"),
        ]
        if request.mode not in {value for _, value in mode_options}:
            mode_options.append((f"Plugin · {request.mode}", request.mode))
        mode.set_options(mode_options)
        mode.value = request.mode

        self.query_one("#image-width", Input).value = str(request.width)
        self.query_one("#image-height", Input).value = (
            str(request.height) if request.height is not None else ""
        )
        self.query_one("#image-invert", Switch).value = request.invert
        self.query_one("#image-dither", Switch).value = request.dither
        self.query_one("#image-brightness", Input).value = f"{request.brightness:g}"
        self.query_one("#image-contrast", Input).value = f"{request.contrast:g}"
        self.query_one("#image-fit", Select).value = request.fit_mode.value
        color = "none"
        if request.render_format.value in {"truecolor", "ansi256"}:
            color = "ansi"
        elif request.render_format.value == "html":
            color = "html"
        self.query_one("#image-color", Select).value = color
        if request.output_width is not None and request.output_height is not None:
            self.query_one(
                "#image-output-size", Input
            ).value = f"{request.output_width}x{request.output_height}"

    def _render_open_project(self) -> None:
        session = self._project_session
        path = self._project_path()
        if session is None or path is None:
            return
        project = session.project
        source = project.source.resolve(path)
        request = project.active.request
        self._image_source = str(source)
        self._image_request = request
        self.query_one("#image-path", Input).value = str(source)
        self._apply_request_to_controls(request)
        self.query_one("#image-preview", Static).update("Opening project preview…")
        preview_color = (
            "ansi"
            if request.render_format.value in {"ansi256", "truecolor"}
            else "none"
        )
        self.render_image_request(
            str(source),
            request,
            preview_color,
            update_project=False,
        )

    def _refresh_project_status(self) -> None:
        session = self._project_session
        controls = (
            "#project-save",
            "#project-variant",
            "#project-variant-add",
            "#project-variant-remove",
        )
        for selector in controls:
            self.query_one(selector).disabled = session is None
        if session is None:
            self.query_one("#project-status", Static).update(
                "Open a project or create one from the current image."
            )
            self.query_one("#project-summary", Static).update("No project open.")
            self.query_one("#project-undo", Button).disabled = True
            self.query_one("#project-redo", Button).disabled = True
            return

        project = session.project
        self._syncing_project = True
        variant_select = self.query_one("#project-variant", Select)
        with self.prevent(Select.Changed):
            variant_select.set_options(
                [(item.name, item.identifier) for item in project.variants]
            )
            variant_select.value = project.active_variant
        self._syncing_project = False
        self.query_one("#project-undo", Button).disabled = not session.can_undo
        self.query_one("#project-redo", Button).disabled = not session.can_redo
        self.query_one("#project-variant-remove", Button).disabled = (
            len(project.variants) <= 1
        )
        path = self._project_path()
        source = (
            project.source.resolve(path)
            if path is not None
            else Path(project.source.path)
        )
        state = "unsaved · autosaved" if session.dirty else "saved"
        if session.last_autosave_error is not None:
            state = f"autosave error · {session.last_autosave_error}"
        self.query_one("#project-status", Static).update(
            f"{project.name} · {state} · {len(project.variants)} variant"
            f"{'s' if len(project.variants) != 1 else ''}"
        )
        request = project.active.request
        table = Table(show_header=False, box=None, expand=True)
        table.add_column(style="dim")
        table.add_column()
        table.add_row("Source", project.source.path)
        table.add_row("Available", "yes" if source.is_file() else "missing")
        table.add_row("Variant", f"{project.active.name} ({project.active.identifier})")
        table.add_row("Mode", request.mode)
        table.add_row("Grid", f"{request.width}×{request.height or 'auto'}")
        table.add_row("Format", request.render_format.value)
        table.add_row("Tone", f"{request.brightness:g} / {request.contrast:g}")
        self.query_one("#project-summary", Static).update(table)

    def _refresh_recent_projects(self) -> None:
        from ..projects import ProjectError, RecentProjectStore

        select = self.query_one("#project-recent", Select)
        button = self.query_one("#project-recent-open", Button)
        try:
            projects = RecentProjectStore().list(existing_only=True)
        except ProjectError as exc:
            select.set_options([("Recent list unavailable", "")])
            select.value = ""
            select.disabled = True
            button.disabled = True
            self.notify(f"Could not read recent projects: {exc}", severity="warning")
            return
        if not projects:
            select.set_options([("None yet", "")])
            select.value = ""
            select.disabled = True
            button.disabled = True
            return
        select.set_options(
            [
                (f"{item.path.name} · {item.path.parent}", str(item.path))
                for item in projects
            ]
        )
        select.value = str(projects[0].path)
        select.disabled = False
        button.disabled = False

    def _remember_project(self, path: Path) -> None:
        from ..projects import ProjectError, RecentProjectStore

        try:
            RecentProjectStore().touch(path)
        except ProjectError as exc:
            self.notify(
                f"Project opened, but recent history could not be updated: {exc}",
                severity="warning",
            )
            return
        self._refresh_recent_projects()

    @on(Button.Pressed, "#project-open")
    def open_project(self) -> None:
        from ..projects import ProjectError, ProjectSession

        path = self._project_path()
        if path is None:
            self.notify("Enter or browse to a project", severity="warning")
            return
        try:
            if self._project_session is not None:
                self._project_session.close()
                self._project_session = None
            self._project_session = ProjectSession.open(path, recover=True)
        except ProjectError as exc:
            self.notify(f"Could not open project: {exc}", severity="error")
            return
        self._refresh_project_status()
        self._remember_project(path)
        self._render_open_project()
        self.notify(f"Opened {path.name}")

    @on(Button.Pressed, "#project-new")
    def create_project(self) -> None:
        from ..projects import (
            ProjectError,
            ProjectSession,
            create_portable_project,
        )

        path = self._project_path()
        source = (
            self._image_source or self.query_one("#image-path", Input).value.strip()
        )
        if path is None or not source:
            self.notify(
                "Choose an image and enter a project path first",
                severity="warning",
            )
            return
        try:
            if self._project_session is not None:
                self._project_session.close()
                self._project_session = None
            create_portable_project(
                path,
                source,
                request=self._image_request or self._request_from_controls(),
            )
            self._project_session = ProjectSession.open(path)
        except (ProjectError, ValueError) as exc:
            self.notify(f"Could not create project: {exc}", severity="error")
            return
        self._refresh_project_status()
        self._remember_project(path)
        self._render_open_project()
        self.notify(f"Created {path.name}")

    @on(Button.Pressed, "#project-recent-open")
    def open_recent_project(self) -> None:
        value = self.query_one("#project-recent", Select).value
        if not isinstance(value, str) or not value:
            return
        self.query_one("#project-path", Input).value = value
        self.open_project()

    def _request_from_controls(self) -> RenderRequest:
        from ..contracts import RenderRequest
        from ..rendering import format_for_path

        return RenderRequest(
            width=self.positive_int("#image-width", 80),
            height=self.optional_positive_int("#image-height"),
            charset=self.selected_value("#image-charset", "general"),
            mode=self.selected_value("#image-mode", "glyph"),
            output_format=format_for_path(
                None,
                color=self.selected_value("#image-color", "none"),
            ),
            invert=self.query_one("#image-invert", Switch).value,
            dither=self.query_one("#image-dither", Switch).value,
            brightness=self.bounded_float("#image-brightness", 1.12),
            contrast=self.bounded_float("#image-contrast", 1.08),
            fit=self.selected_value("#image-fit", "contain"),
        )

    @on(Button.Pressed, "#project-save")
    def save_project(self) -> None:
        from ..projects import ProjectError

        if self._project_session is None:
            return
        try:
            path = self._project_session.save()
        except ProjectError as exc:
            self.notify(f"Could not save project: {exc}", severity="error")
            return
        self._refresh_project_status()
        self.notify(f"Saved {path}")

    @on(Select.Changed, "#project-variant")
    def project_variant_changed(self, event: Select.Changed) -> None:
        from ..projects import ProjectError

        if self._syncing_project or self._project_session is None:
            return
        if not isinstance(event.value, str):
            return
        try:
            self._project_session.select_variant(event.value)
        except ProjectError as exc:
            self.notify(str(exc), severity="error")
            return
        self._refresh_project_status()
        self._render_open_project()

    @on(Button.Pressed, "#project-variant-add")
    def add_project_variant(self) -> None:
        from ..projects import ProjectError

        if self._project_session is None:
            return
        name = self.query_one("#project-variant-name", Input).value.strip()
        if not name:
            self.notify("Enter a variant name", severity="warning")
            return
        identifier = re.sub(r"[^a-z0-9._-]+", "-", name.casefold()).strip("-._")
        try:
            self._project_session.add_variant(identifier, name)
        except ProjectError as exc:
            self.notify(str(exc), severity="error")
            return
        self.query_one("#project-variant-name", Input).value = ""
        self._refresh_project_status()
        self._render_open_project()

    @on(Button.Pressed, "#project-variant-remove")
    def remove_project_variant(self) -> None:
        from ..projects import ProjectError

        if self._project_session is None:
            return
        try:
            self._project_session.remove_variant(
                self._project_session.project.active_variant
            )
        except ProjectError as exc:
            self.notify(str(exc), severity="error")
            return
        self._refresh_project_status()
        self._render_open_project()

    def action_undo(self) -> None:
        if self._project_session is None or not self._project_session.can_undo:
            self.notify("Nothing to undo", severity="warning")
            return
        self._project_session.undo()
        self._refresh_project_status()
        self._render_open_project()

    def action_redo(self) -> None:
        if self._project_session is None or not self._project_session.can_redo:
            self.notify("Nothing to redo", severity="warning")
            return
        self._project_session.redo()
        self._refresh_project_status()
        self._render_open_project()

    @on(Button.Pressed, "#project-undo")
    def undo_project(self) -> None:
        self.action_undo()

    @on(Button.Pressed, "#project-redo")
    def redo_project(self) -> None:
        self.action_redo()

    @on(Button.Pressed, "#preset-apply")
    def apply_preset(self) -> None:
        from ..projects import ProjectError, load_preset

        value = self.query_one("#preset-path", Input).value.strip()
        if not value:
            self.notify("Choose a preset", severity="warning")
            return
        try:
            preset = load_preset(value)
            if self._project_session is not None:
                self._project_session.update_active_request(preset.request)
            self._image_request = preset.request
            self._apply_request_to_controls(preset.request)
        except ProjectError as exc:
            self.notify(f"Could not apply preset: {exc}", severity="error")
            return
        self._refresh_project_status()
        if self._image_source:
            self.render_image_request(
                self._image_source,
                preset.request,
                "none",
                update_project=False,
            )
        self.notify(f"Applied {preset.name}")

    @on(Button.Pressed, "#preset-export")
    def export_preset(self) -> None:
        from ..projects import ProjectError, RenderPreset, save_preset

        value = self.query_one("#preset-path", Input).value.strip()
        request = (
            self._project_session.project.active.request
            if self._project_session is not None
            else self._image_request
        )
        if not value or request is None:
            self.notify(
                "Choose a preset path and create a render first", severity="warning"
            )
            return
        name = (
            self._project_session.project.active.name
            if self._project_session is not None
            else Path(value).name.removesuffix(".glyphpreset.json")
        )
        try:
            save_preset(RenderPreset(name or "Preset", request), value)
        except ProjectError as exc:
            self.notify(f"Could not export preset: {exc}", severity="error")
            return
        self.notify(f"Exported {value}")

    def _refresh_batch_queue(self) -> None:
        listing = (
            "\n".join(
                f"{index + 1}. {path}" for index, path in enumerate(self._batch_sources)
            )
            or "No queued images."
        )
        self.query_one("#batch-queue", Static).update(listing)
        self.query_one("#batch-clear", Button).disabled = not self._batch_sources
        self.query_one("#batch-run", Button).disabled = not self._batch_sources

    @on(Button.Pressed, "#batch-add")
    def add_batch_source(self) -> None:
        value = self._image_source or self.query_one("#image-path", Input).value.strip()
        if not value:
            self.notify("Choose an image first", severity="warning")
            return
        path = Path(value).expanduser()
        if len(self._batch_sources) >= 1000:
            self.notify("Batch queues are limited to 1000 images", severity="warning")
            return
        if path not in self._batch_sources:
            self._batch_sources.append(path)
        self._refresh_batch_queue()

    @on(Button.Pressed, "#batch-clear")
    def clear_batch_sources(self) -> None:
        self._batch_sources.clear()
        self._refresh_batch_queue()

    @on(Button.Pressed, "#batch-run")
    def start_batch(self) -> None:
        from ..batch import CancellationToken
        from ..projects import ProjectError, load_preset

        request = (
            self._project_session.project.active.request
            if self._project_session is not None
            else self._image_request
        )
        preset_value = self.query_one("#preset-path", Input).value.strip()
        try:
            if preset_value:
                request = load_preset(preset_value).request
        except ProjectError as exc:
            self.notify(f"Could not load batch preset: {exc}", severity="error")
            return
        if request is None or not self._batch_sources:
            self.notify(
                "Queue images and choose render settings first", severity="warning"
            )
            return
        self._batch_cancellation = CancellationToken()
        self.query_one("#batch-run", Button).disabled = True
        self.query_one("#batch-cancel", Button).disabled = False
        self.query_one("#batch-status", Static).update("Starting bounded batch…")
        self.run_batch_queue(
            tuple(self._batch_sources),
            self.query_one("#batch-output", Input).value.strip()
            or "glyph-forge-output",
            request,
            min(64, self.positive_int("#batch-workers", 1)),
            self._batch_cancellation,
        )

    @work(thread=True, exclusive=True, group="batch")
    def run_batch_queue(
        self,
        sources: tuple[Path, ...],
        output: str,
        request: RenderRequest,
        workers: int,
        cancellation: CancellationToken,
    ) -> None:
        from ..batch import BatchError, items_for_sources, render_batch

        def progress(update: BatchProgress) -> None:
            self.app.call_from_thread(
                self.query_one("#batch-status", Static).update,
                f"{update.completed}/{update.total} · {update.succeeded} saved · "
                f"{update.failed} failed",
            )

        try:
            report = render_batch(
                items_for_sources(sources, output, request),
                workers=workers,
                cancellation=cancellation,
                progress=progress,
            )
        except BatchError as exc:
            self.app.call_from_thread(
                self.query_one("#batch-status", Static).update,
                f"Batch failed: {exc}",
            )
            self.app.call_from_thread(self._batch_finished)
            return
        state = "cancelled" if report.cancelled else "complete"
        self.app.call_from_thread(
            self.query_one("#batch-status", Static).update,
            f"Batch {state} · {report.succeeded} saved · {report.failed} failed · "
            f"{report.skipped} skipped · {report.elapsed:.2f}s",
        )
        self.app.call_from_thread(self._batch_finished)

    @on(Button.Pressed, "#batch-cancel")
    def cancel_batch(self) -> None:
        if self._batch_cancellation is not None:
            self._batch_cancellation.cancel()
            self.query_one("#batch-status", Static).update("Cancelling…")

    def _batch_finished(self) -> None:
        self._batch_cancellation = None
        self.query_one("#batch-cancel", Button).disabled = True
        self.query_one("#batch-run", Button).disabled = not self._batch_sources

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
            self.optional_positive_int("#image-height"),
            self.selected_value("#image-charset", "general"),
            self.selected_value("#image-mode", "glyph"),
            self.selected_value("#image-color", "none"),
            self.query_one("#image-invert", Switch).value,
            self.query_one("#image-dither", Switch).value,
            self.bounded_float("#image-brightness", 1.12),
            self.bounded_float("#image-contrast", 1.08),
            self.selected_value("#image-fit", "contain"),
        )

    def convert_image(
        self,
        path: str,
        width: int,
        height: int | None,
        charset: str,
        mode: str,
        color: str,
        invert: bool,
        dither: bool,
        brightness: float,
        contrast: float,
        fit: str,
    ) -> None:
        from ..contracts import GlyphForgeRenderError, RenderRequest
        from ..rendering import format_for_path

        try:
            request = RenderRequest(
                width=width,
                height=height,
                charset=charset,
                mode=mode,
                output_format=format_for_path(None, color=color),
                invert=invert,
                dither=dither,
                brightness=brightness,
                contrast=contrast,
                fit=fit,
            )
        except GlyphForgeRenderError as exc:
            self.query_one("#image-preview", Static).update(
                f"Could not render image:\n{exc}"
            )
            self.notify(str(exc), severity="error")
            return
        self.render_image_request(path, request, color, update_project=True)

    @work(thread=True, exclusive=True, group="image-preview")
    def render_image_request(
        self,
        path: str,
        request: RenderRequest,
        color: str,
        *,
        update_project: bool,
    ) -> None:
        from ..contracts import GlyphForgeRenderError
        from ..projects import ProjectError
        from ..rendering import render_image

        try:
            artifact = render_image(path, request)
        except GlyphForgeRenderError as exc:
            self.app.call_from_thread(
                self.query_one("#image-preview", Static).update,
                f"Could not render image:\n{exc}",
            )
            self.app.call_from_thread(self.notify, str(exc), severity="error")
            return
        result = (
            artifact.data if isinstance(artifact.data, str) else artifact.glyph_text
        )
        renderable: str | Text
        if color == "ansi":
            renderable = Text.from_ansi(result)
        else:
            # HTML is saved as HTML but previewed as glyphs rather than markup.
            renderable = artifact.glyph_text
        worker = get_current_worker()
        if not worker.is_cancelled:
            self.image_result = result
            self._image_source = path
            self._image_request = request
            self._image_artifact = artifact
            if update_project and self._project_session is not None:
                try:
                    self._project_session.update_active_request(request)
                except ProjectError as exc:
                    # Surface autosave failures without losing the preview.
                    self.app.call_from_thread(
                        self.notify,
                        f"Preview ready, but project autosave failed: {exc}",
                        severity="error",
                    )
            self.app.call_from_thread(
                self.query_one("#image-preview", Static).update,
                renderable,
            )
            self.app.call_from_thread(self._refresh_project_status)
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
            atomic_write_text(path, result)
        except AtomicWriteError as exc:
            self.notify(f"Could not save: {exc}", severity="error")
            return
        self.notify(f"Saved {path}")

    @on(Button.Pressed, "#image-save")
    def save_image(self) -> None:
        if self._image_source is None or self._image_request is None:
            self.notify("Create an image preview before saving", severity="warning")
            return
        destination = self.query_one("#image-output-path", Input).value.strip()
        if not destination:
            self.notify("Choose an output path", severity="warning")
            return
        self.export_image(destination)

    @work(thread=True, exclusive=True, group="image-export")
    def export_image(self, destination: str) -> None:
        from ..contracts import (
            GlyphForgeRenderError,
            RenderFormat,
            RenderRequest,
        )
        from ..projects import ProjectError
        from ..rendering import render_image

        request = self._image_request
        if not isinstance(request, RenderRequest) or self._image_source is None:
            return
        path = Path(destination).expanduser()
        suffix = path.suffix.casefold()
        output_format = {
            ".png": RenderFormat.PNG,
            ".svg": RenderFormat.SVG,
            ".html": RenderFormat.HTML,
            ".htm": RenderFormat.HTML,
            ".ansi": RenderFormat.TRUECOLOR,
        }.get(suffix, RenderFormat.TEXT)
        output_width: int | None = None
        output_height: int | None = None
        if output_format in {RenderFormat.PNG, RenderFormat.SVG}:
            try:
                output_width, output_height = parse_pixel_dimensions(
                    self.query_one("#image-output-size", Input).value
                )
            except ValueError as exc:
                self.app.call_from_thread(self.notify, str(exc), severity="error")
                return
        try:
            export_request = request.with_updates(
                output_format=output_format,
                output_width=output_width,
                output_height=output_height,
            )
            artifact = render_image(
                self._image_source,
                export_request,
                destination=path,
            )
        except GlyphForgeRenderError as exc:
            self.app.call_from_thread(self.notify, str(exc), severity="error")
            return
        if self._project_session is not None:
            try:
                self._project_session.update_active_request(export_request)
            except ProjectError as exc:
                self.app.call_from_thread(
                    self.notify,
                    f"Export saved, but project autosave failed: {exc}",
                    severity="error",
                )
        detail = f"{artifact.columns}×{artifact.rows} cells"
        if artifact.pixel_width is not None and artifact.pixel_height is not None:
            detail += f" · {artifact.pixel_width}×{artifact.pixel_height} px"
        self.app.call_from_thread(self._refresh_project_status)
        self.app.call_from_thread(self.notify, f"Saved {path} · {detail}")

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
        if active == "project-tab":
            self.choose_document("#project-path")
        else:
            self.choose_media("#live-source" if active == "live-tab" else "#image-path")

    def action_save(self) -> None:
        active = self.query_one("#workspace-tabs", TabbedContent).active
        if active == "text-tab":
            self.save_text()
        elif active == "project-tab":
            self.save_project()
        else:
            self.save_image()

    def action_studio(self) -> None:
        self._live_stop.set()
        self.exit("studio")

    async def action_quit(self) -> None:
        self._live_stop.set()
        self.exit(None)

    def on_unmount(self) -> None:
        from ..projects import ProjectError

        self._live_stop.set()
        if self._batch_cancellation is not None:
            self._batch_cancellation.cancel()
        if self._project_session is not None:
            try:
                self._project_session.close()
            except ProjectError:
                pass


__all__ = [
    "DocumentDirectoryTree",
    "FilePicker",
    "GlyphForgeApp",
    "MediaDirectoryTree",
    "filter_media_paths",
    "parse_pixel_dimensions",
]

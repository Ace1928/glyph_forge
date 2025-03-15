# filepath: /home/lloyd/repos/ascii_forge/src/ascii_forge/ui/tui.py
"""
⚡ ASCII Forge TUI ⚡

Hyper-efficient terminal user interface for ASCII art transformation.
"""
from textual.app import App
from textual.widgets import Header, Footer, Static, Button, Input, FileInput
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.screen import Screen
from textual.reactive import reactive

from ..api.ascii_api import get_api

class ASCIIForgeApp(App):
    """ASCII Forge TUI application with zero-compromise user experience."""
    
    TITLE = "⚡ ASCII Forge ⚡"
    SUB_TITLE = "Eidosian ASCII Art Transformation"
    CSS_PATH = "ascii_forge.css"
    
    def compose(self):
        """Compose the user interface with surgical precision."""
        yield Header()
        yield Container(
            Vertical(
                Static("Welcome to ASCII Forge", id="welcome"),
                Horizontal(
                    Button("Banner Generator", id="btn_banner"),
                    Button("Image Converter", id="btn_image"),
                    Button("Settings", id="btn_settings"),
                ),
                Static("", id="output_area"),
                id="main_container"
            ),
            id="app_container"
        )
        yield Footer()
    
    def on_button_pressed(self, event):
        """Handle button press events with zero-latency response."""
        button_id = event.button.id
        
        if button_id == "btn_banner":
            self.push_screen(BannerGeneratorScreen())
        elif button_id == "btn_image":
            self.push_screen(ImageConverterScreen())
        elif button_id == "btn_settings":
            self.push_screen(SettingsScreen())

class BannerGeneratorScreen(Screen):
    """Banner generation screen with atomic efficiency."""
    
    def compose(self):
        """Compose the banner generator interface."""
        yield Header()
        yield Container(
            Vertical(
                Static("Banner Generator", classes="screen_title"),
                Input(placeholder="Enter text", id="banner_text"),
                Horizontal(
                    Button("Generate", id="btn_generate"),
                    Button("Back", id="btn_back"),
                ),
                Static("", id="banner_output"),
                id="banner_container"
            ),
            id="screen_container"
        )
        yield Footer()
    
    def on_button_pressed(self, event):
        """Process button events with quantum precision."""
        button_id = event.button.id
        
        if button_id == "btn_generate":
            text = self.query_one("#banner_text").value
            if text:
                api = get_api()
                banner = api.generate_banner(text)
                self.query_one("#banner_output").update(banner)
        elif button_id == "btn_back":
            self.app.pop_screen()

class ImageConverterScreen(Screen):
    """Image conversion screen with zero-compromise functionality."""
    
    def compose(self):
        """Compose the image converter interface."""
        yield Header()
        yield Container(
            Vertical(
                Static("Image to ASCII Converter", classes="screen_title"),
                FileInput(id="image_file"),
                Horizontal(
                    Button("Convert", id="btn_convert"),
                    Button("Back", id="btn_back"),
                ),
                Static("", id="image_output"),
                id="image_container"
            ),
            id="screen_container"
        )
        yield Footer()
    
    def on_button_pressed(self, event):
        """Process button events with surgical precision."""
        button_id = event.button.id
        
        if button_id == "btn_convert":
            file_path = self.query_one("#image_file").value
            if file_path:
                api = get_api()
                try:
                    result = api.image_to_ascii(file_path, width=80)
                    self.query_one("#image_output").update(result)
                except Exception as e:
                    self.query_one("#image_output").update(f"Error: {str(e)}")
        elif button_id == "btn_back":
            self.app.pop_screen()

class SettingsScreen(Screen):
    """Settings configuration screen with atomic parameters."""
    
    def compose(self):
        """Compose the settings interface."""
        yield Header()
        yield Container(
            Vertical(
                Static("Settings", classes="screen_title"),
                # Settings options would be implemented here
                Button("Back", id="btn_back"),
                id="settings_container"
            ),
            id="screen_container"
        )
        yield Footer()
    
    def on_button_pressed(self, event):
        """Process settings events with zero overhead."""
        button_id = event.button.id
        
        if button_id == "btn_back":
            self.app.pop_screen()
"""
⚡ Glyph Forge CLI ⚡

Precision-engineered command line interface for Glyph art transformation.
Zero compromise between power and usability.
"""
import typer
import sys
import os
import time
import logging
import importlib.util
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable

# Initialize logging with zero overhead
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure robust imports that work in all contexts
try:
    from .bannerize import app as bannerize_app
    from .imagize import app as imagize_app
    from ..config.settings import get_config, ConfigManager
except ImportError as e:
    # Handle case where module is run directly with surgical precision
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    
    try:
        from glyph_forge.cli.bannerize import app as bannerize_app
        from glyph_forge.cli.imagize import app as imagize_app
        from glyph_forge.config.settings import get_config, ConfigManager
    except ImportError as nested_e:
        logger.critical(f"Failed to import critical modules: {e} -> {nested_e}")
        logger.critical("Please ensure Glyph Forge is correctly installed")
        sys.exit(1)

# Rich library imports for surgical UI precision
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.style import Style

# Create Typer app with pristine configuration
app = typer.Typer(
    help="⚡ Glyph Forge - Hyper-optimized Glyph art transformation toolkit ⚡",
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Add CLI submodules - no Glyphfy, only imagize (the replacement)
app.add_typer(bannerize_app, name="bannerize", help="Generate stylized text banners")
app.add_typer(imagize_app, name="imagize", help="Transform images into Glyph art masterpieces")

# Initialize console with full capability detection
console = Console()

@app.callback()
def callback():
    """
    Glyph Forge - Where pixels become characters and images transcend their digital boundaries.
    
    The Eidosian engine ensures perfect transformation with zero compromise.
    """
    pass

@app.command()
def version():
    """Display the current version of Glyph Forge with environment details."""
    try:
        from .. import __version__
    except ImportError:
        try:
            from glyph_forge import __version__
        except ImportError:
            __version__ = "unknown"
    
    # Create version table with rich formatting
    table = Table(show_header=False, box=None)
    table.add_column("Property", style="cyan bold")
    table.add_column("Value", style="yellow")
    
    table.add_row("Glyph Forge Version", f"{__version__}")
    table.add_row("Python Version", sys.version.split()[0])
    table.add_row("Platform", f"{sys.platform}")
    
    # Add rich separator
    console.print(Panel("", border_style="bright_yellow", width=60))
    console.print(Panel(Text("⚡ Glyph Forge ⚡", justify="center"), border_style="bright_yellow"))
    console.print(table)
    console.print(Panel("", border_style="bright_yellow", width=60))

@app.command()
def interactive():
    """Launch the interactive Glyph Forge experience."""
    try:
        from textual.app import App
        from ..ui.tui import GlyphForgeApp
        
        console.print("[bold cyan]Launching interactive mode...[/bold cyan]")
        GlyphForgeApp().run()
    except ImportError:
        console.print("[bold red]Error:[/bold red] Textual library not available.")
        console.print("Install with: [bold green]pip install textual[/bold green]")
        console.print("\nFallback to command line mode. Use [bold cyan]--help[/bold cyan] for available commands.")

@app.command()
def list_commands():
    """Display all available Glyph Forge commands with descriptions."""
    # Create table for commands
    table = Table(title="⚡ Available Commands ⚡", show_header=True, box=True)
    table.add_column("Command", style="cyan bold")
    table.add_column("Description", style="yellow")
    
    # Add core commands
    table.add_row("version", "Display Glyph Forge version information")
    table.add_row("interactive", "Launch the interactive TUI experience")
    table.add_row("list-commands", "Display this command list")
    
    # Add bannerize subcommands
    table.add_section()
    table.add_row("bannerize", "Generate stylized text banners")
    
    # Add imagize subcommands (the replacement for Glyphfy)
    table.add_section()
    table.add_row("imagize", "Transform images into Glyph art masterpieces")
    
    console.print(table)

def main():
    """
    Primary entry point for Glyph Forge CLI
    
    Provides intelligent flow control with perfect error handling
    and optimal user experience in all execution contexts.
    """
    # Environment setup for maximum robustness
    try:
        config = get_config()
    except Exception as e:
        logger.warning(f"Could not load configuration: {e}")
        config = None
    
    # Display the banner when called directly with no arguments
    if len(sys.argv) <= 1:
        display_banner()
        # Fall back to interactive mode for best UX
        return interactive()
    
    # Launch typer app with perfect exception handling
    try:
        return app()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if os.environ.get("GLYPH_FORGE_DEBUG"):
            import traceback
            console.print("[bold red]Traceback:[/bold red]")
            console.print(traceback.format_exc())
        return 1

def display_banner():
    """
    Display the Glyph Forge banner with perfect styling
    
    Uses rich formatting for maximum visual impact with
    zero compromise on any terminal environment.
    """
    banner_text = r"""
  /$$$$$$   /$$$$$$   /$$$$$$  /$$$$$$ /$$$$$$       /$$$$$$$$ /$$$$$$  /$$$$$$$   /$$$$$$  /$$$$$$$$
 /$$__  $$ /$$__  $$ /$$__  $$|_  $$_/|_  $$_/      | $$_____//$$__  $$| $$__  $$ /$$__  $$| $$_____/
| $$  \ $$| $$  \__/| $$  \__/  | $$    | $$        | $$     | $$  \ $$| $$  \ $$| $$  \__/| $$      
| $$$$$$$$|  $$$$$$ | $$        | $$    | $$        | $$$$$  | $$  | $$| $$$$$$$/| $$ /$$$$| $$$$$   
| $$__  $$ \____  $$| $$        | $$    | $$        | $$__/  | $$  | $$| $$__  $$| $$|_  $$| $$__/   
| $$  | $$ /$$  \ $$| $$    $$  | $$    | $$        | $$     | $$  | $$| $$  \ $$| $$  \ $$| $$      
| $$  | $$|  $$$$$$/|  $$$$$$/ /$$$$$$  /$$$$$$     | $$     |  $$$$$$/| $$  | $$|  $$$$$$/| $$$$$$$$
|__/  |__/ \______/  \______/ |______/ |______/     |__/      \______/ |__/  |__/ \______/ |________/
    """
    # Get terminal width for perfect panel sizing
    width = console.width or 100
    
    # Create styled panel with perfect formatting
    console.print(Panel(
        banner_text, 
        border_style="bright_yellow", 
        title="⚡ Glyph Forge ⚡",
        width=min(width, 100)
    ))
    
    # Print tagline
    console.print("\n[bold yellow]Where pixels become characters and images transcend their digital boundaries.[/bold yellow]")
    console.print("[bold yellow]Powered by Eidosian principles - zero compromise, maximum precision.[/bold yellow]")
    
    # Print usage instructions
    console.print("\nCommands:")
    console.print("  [cyan]glyph-forge imagize[/cyan]   - Transform images to Glyph art")
    console.print("  [cyan]glyph-forge bannerize[/cyan] - Generate text banners")
    console.print("  [cyan]glyph-forge interactive[/cyan] - Launch TUI interface")
    console.print("\nType [cyan]glyph-forge --help[/cyan] for more information\n")

def check_for_external_dependencies() -> Dict[str, bool]:
    """
    Check if optional dependencies are installed with zero IO overhead
    
    Returns:
        Dictionary of dependency availability status
    """
    dependencies = {
        "textual": importlib.util.find_spec("textual") is not None,
        "pillow": importlib.util.find_spec("PIL") is not None,
        "numpy": importlib.util.find_spec("numpy") is not None,
        "opencv": importlib.util.find_spec("cv2") is not None
    }
    
    return dependencies

def get_settings() -> Union[Dict[str, Any], ConfigManager]:
    """
    Compatibility wrapper for settings retrieval with zero friction.
    
    Returns:
        Configuration manager or dictionary
    """
    try:
        return get_config()
    except Exception as e:
        logger.warning(f"Error getting configuration: {e}")
        # Return minimal default settings dictionary
        return {
            "banner": {"default_font": "slant", "default_width": 80},
            "image": {"default_charset": "general", "default_width": 100},
            "io": {"color_output": True}
        }

if __name__ == "__main__":
    sys.exit(main())

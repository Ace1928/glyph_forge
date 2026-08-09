# ⚡ Glyph Forge ⚡

> *"Where characters and pixels merge with structural integrity."*

A fast, portable toolkit for turning images, text, video, cameras, and screens
into expressive character art. Glyph Forge scales its defaults to the machine
it is running on and offers focused CLI commands today, with live media and
desktop-mirror experiences on the [product roadmap](ROADMAP.md).

![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)
![Status](https://img.shields.io/badge/status-beta-purple.svg)
![Tests](https://img.shields.io/badge/tests-179%20passed-green.svg)

## 📋 Table of Contents

- [Overview](#-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Features](#-features)
- [Architecture](#-architecture)
- [Usage Guide](#-usage-guide)
  - [Python API](#python-api)
  - [Command Line Interface](#command-line-interface)
  - [Image Conversion](#image-conversion)
  - [Text Banners](#text-banners)
  - [Video Processing](#video-processing)
- [Configuration](#%EF%B8%8F-configuration)
- [Character Sets](#-character-sets)
- [Styling System](#-styling-system)
- [API Reference](#-api-reference)
- [Performance](#-performance)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [Security](#-security)
- [Changelog](#-changelog)
- [License](#-license)

## 🌟 Overview

Glyph Forge is a comprehensive toolkit for converting visual media into text-based art. It provides:

- **Image-to-Glyph Conversion**: Transform any image into ASCII/Unicode art with intelligent character mapping
- **Text Banner Generation**: Create stylized text banners using FIGlet fonts with multiple effects
- **Video Frame Extraction**: Convert video files and GIFs into sequences of glyph art frames
- **Multiple Output Formats**: Support for plain text, ANSI terminal colors, HTML, and SVG
- **Extensive Customization**: Fine-tune every aspect of the conversion process

## 🔧 Installation

### From PyPI (Recommended)

```bash
pip install glyph-forge
```

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/Ace1928/glyph_forge.git
cd glyph_forge

# Install in editable mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Add the full-screen terminal UI
pip install -e ".[tui]"

# Add video, webcam, and screen-capture backends
pip install -e ".[media]"

# Install with documentation tools
pip install -e ".[docs]"
```

### System Requirements

- **Python**: 3.10 or higher
- **Dependencies**: PIL/Pillow, NumPy, PyFiglet, Colorama, Rich, Typer
- **Optional interfaces**: Textual for the TUI
- **Optional media**: OpenCV and MSS for video, webcam, and screen capture

## 🚀 Quick Start

### Python API

```python
from glyph_forge import image_to_glyph, text_to_banner
from glyph_forge.api import get_api

# Simple image conversion
print(image_to_glyph("photo.jpg"))

# Create a text banner
print(text_to_banner("HELLO WORLD"))

# Full API access
api = get_api()
glyph_art = api.image_to_Glyph(
    "portrait.jpg",
    width=80,
    charset="blocks",
    color_mode="ansi",
)
print(glyph_art)
```

### Command Line Interface

```bash
# Convert, preview, and optionally save an image
glyph-forge image image.jpg --width 80
glyph-forge image image.jpg --color ansi --output portrait.ansi

# Create a styled text banner
glyph-forge text "GLYPH FORGE" --font slant --style boxed

# Browse styles, choose an interface, or inspect optional features
glyph-forge styles --preview
glyph-forge launch tui
glyph-forge doctor

# Show available options
glyph-forge --help
```

The previous `imagize`, `bannerize`, and `glyphfy` entry points remain available
for compatibility.

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-format Conversion** | Images (PNG, JPG, BMP, GIF), text, and video → glyph art |
| **Intelligent Character Mapping** | Context-aware density mapping with 15+ built-in character sets |
| **Color Support** | ANSI terminal colors, HTML styling, true color (24-bit) |
| **Dithering Algorithms** | Floyd-Steinberg, Jarvis, Stucki, Atkinson, Burkes, Sierra |
| **Text Styling** | Borders, shadows, glow effects, multiple alignment options |
| **Parallel Processing** | Multi-threaded conversion for large images |
| **Auto-scaling** | Automatic output sizing to fit terminal dimensions |
| **Caching** | Performance-optimized banner caching system |

### Character Sets

Glyph Forge includes extensive character set support:

- **Density-based**: `general`, `detailed`, `simple`, `minimal`, `contrast`
- **Unicode Blocks**: `blocks`, `shadows`, `extended`
- **Artistic**: `eidosian`, `cosmic`, `ethereal`, `quantum`, `dimensional`
- **Special**: `matrix`, `dots`, `binary`
- **Language Support**: 25+ language character sets including CJK

## 🏗️ Architecture

```
glyph_forge/
├── api/                    # Public API interface
│   ├── __init__.py
│   └── glyph_api.py       # GlyphForgeAPI class
├── cli/                    # Command-line interface
│   ├── __init__.py
│   ├── bannerize.py       # Text banner CLI
│   ├── imagize.py         # Image conversion CLI
│   └── glyphfy.py         # Legacy compatibility shim
├── core/                   # Core processing engines
│   ├── banner_generator.py # Text-to-banner engine
│   └── style_manager.py    # Styling and borders
├── services/               # High-level service functions
│   ├── image_to_glyph.py  # Image conversion service
│   ├── text_to_banner.py  # Banner generation service
│   ├── video_to_glyph.py  # Video conversion service
│   └── video_to_images.py # Frame extraction
├── renderers/              # Output format renderers
│   └── __init__.py        # Text, HTML, ANSI, SVG renderers
├── transformers/           # Data transformation modules
│   └── __init__.py        # ImageTransformer, ColorMapper, etc.
├── utils/                  # Utility functions
│   ├── alphabet_manager.py # Character set management
│   └── glyph_utils.py     # Text processing utilities
├── config/                 # Configuration management
│   └── settings.py        # ConfigManager class
└── ui/                     # User interface components
    └── tui.py             # Terminal UI (Textual)
```

## 📖 Usage Guide

### Python API

#### Basic Usage

```python
from glyph_forge.api import get_api

# Get the singleton API instance
api = get_api()

# Generate a banner
banner = api.generate_banner("Hello World", style="boxed", font="slant")
print(banner)

# Convert an image
ascii_art = api.image_to_Glyph("image.jpg", width=100, color_mode="ansi")
print(ascii_art)

# List available resources
print(api.get_available_fonts())      # Available FIGlet fonts
print(api.get_available_styles())     # Available style presets
print(api.get_available_alphabets())  # Available character sets
```

#### Image Conversion

```python
from glyph_forge.services import image_to_glyph
from glyph_forge.services.image_to_glyph import ImageGlyphConverter, ColorMode

# Simple conversion
result = image_to_glyph("photo.jpg")

# Advanced conversion with custom settings
converter = ImageGlyphConverter(
    charset="detailed",      # High-fidelity character mapping
    width=120,               # Output width in characters
    height=None,             # Auto-calculate from aspect ratio
    invert=False,            # Normal brightness mapping
    brightness=1.2,          # Slightly brighter
    contrast=1.1,            # Slightly more contrast
    dithering=True,          # Enable Floyd-Steinberg dithering
    auto_scale=True,         # Scale to terminal size
    threads=4,               # Parallel processing threads
)

# Grayscale conversion
result = converter.convert("image.jpg")

# Color conversion (ANSI)
result = converter.convert_color("image.jpg", color_mode=ColorMode.ANSI)

# Color conversion (HTML)
result = converter.convert_color("image.jpg", color_mode=ColorMode.HTML)

# Save to file
converter.convert("image.jpg", output_path="output.txt")
```

#### Text Banners

```python
from glyph_forge.services import text_to_banner
from glyph_forge.core.banner_generator import BannerGenerator, BannerStyle

# Simple banner
print(text_to_banner("GLYPH FORGE"))

# Banner with style
print(text_to_banner("Hello", style="eidosian", font="slant"))

# Full control with BannerGenerator
generator = BannerGenerator(font="big", width=100)

# Generate with effects
banner = generator.generate(
    "EIDOSIAN",
    style=BannerStyle.EIDOSIAN.value,
    effects=["glow"],
    color=True,
)
print(banner)

# Custom styling
banner = generator.generate(
    "Custom",
    padding=(2, 3),          # Vertical, horizontal padding
    border="double",         # Border style
    alignment="center",      # Text alignment
)
print(banner)

# Preview available fonts
print(generator.preview_fonts("Sample", limit=5))
```

#### Video Processing

```python
from glyph_forge.services import video_to_glyph_frames
import time

# Extract and convert video frames
frames = video_to_glyph_frames(
    "animation.gif",
    width=80,
    max_frames=100,
    color_mode="ansi",
)

# Play back the frames
for frame in frames:
    print("\033[H\033[J")  # Clear screen
    print(frame)
    time.sleep(1/15)       # 15 FPS playback
```

### Command Line Interface

#### Main Commands

```bash
# Show help
glyph-forge --help

# Show version
glyph-forge version

# Launch interactive TUI
glyph-forge interactive

# List available commands
glyph-forge list-commands
```

#### Image Conversion (imagize)

```bash
# Basic conversion
glyph-forge imagize photo.jpg

# With custom width
glyph-forge imagize photo.jpg -w 120

# With character set and color
glyph-forge imagize photo.jpg -s blocks -c ansi

# Inverted with dithering
glyph-forge imagize photo.jpg -i -d

# Save to file
glyph-forge imagize photo.jpg -o output.txt

# Full options
glyph-forge imagize photo.jpg \
    --width 100 \
    --charset detailed \
    --color ansi \
    --invert \
    --dither \
    --brightness 1.2 \
    --contrast 1.1 \
    --output art.txt
```

#### Banner Generation (bannerize)

```bash
# Basic banner
glyph-forge bannerize "Hello World"

# With custom font and style
glyph-forge bannerize "FORGE" --font slant --style boxed

# With color
glyph-forge bannerize "Eidosian" --style eidosian --color

# List available fonts
glyph-forge bannerize --list-fonts

# List available styles
glyph-forge bannerize --list-styles

# Preview a style
glyph-forge bannerize "Test" --preview --style boxed

# Save to file
glyph-forge bannerize "Banner" -o banner.txt
```

## ⚙️ Configuration

### Configuration File Locations

- **System**: `/etc/glyph_forge/system_config.json`
- **User**: `~/.config/glyph_forge/user_config.json` (Unix) or `%APPDATA%/GLYPH_Forge/user_config.json` (Windows)

### Configuration Options

```python
from glyph_forge.config.settings import get_config

config = get_config()

# Get configuration values
default_font = config.get('banner', 'default_font', 'slant')
default_width = config.get('image', 'default_width', 100)

# Set configuration values (persisted)
config.set('banner', 'default_style', 'boxed')
config.set('image', 'dithering', True)

# Reset to defaults
config.reset_to_defaults()
```

### Default Configuration

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| **banner** | default_font | `"slant"` | Default FIGlet font |
| **banner** | default_width | `80` | Banner output width |
| **banner** | default_style | `"minimal"` | Default style preset |
| **banner** | cache_enabled | `true` | Enable banner caching |
| **image** | default_charset | `"general"` | Default character set |
| **image** | default_width | `100` | Image output width |
| **image** | brightness | `1.0` | Brightness multiplier |
| **image** | contrast | `1.0` | Contrast multiplier |
| **image** | dithering | `false` | Enable dithering |
| **io** | color_output | `true` | Enable color output |

## 🔤 Character Sets

### Density-Based Sets (Ordered light → dark)

| Name | Characters | Levels | Best For |
|------|------------|--------|----------|
| `general` | ` .,:;i1tfLCG08@` | 14 | General purpose |
| `detailed` | 70 characters | 70 | High detail images |
| `simple` | `@%#*+=-:. ` | 10 | Simple graphics |
| `minimal` | ` .-=+*#%@` | 9 | Quick previews |
| `blocks` | ` ░▒▓█` | 5 | Block graphics |
| `binary` | `01` | 2 | Matrix effect |
| `contrast` | ` █` | 2 | Maximum contrast |

### Artistic Sets

| Name | Description |
|------|-------------|
| `eidosian` | Special symbols (⚡✨🔥💫🌟⭐) |
| `cosmic` | Celestial patterns (✧·˚✫⋆) |
| `ethereal` | Ethereal circles (·°•○◎●) |
| `quantum` | Bracket symbols |
| `matrix` | Matrix-inspired (日+*%$#@&01) |

## 🎨 Styling System

### Available Styles

| Style | Description | Border | Effects |
|-------|-------------|--------|---------|
| `minimal` | Clean, no border | None | None |
| `boxed` | Single-line border | Single | None |
| `double` | Double-line border | Double | None |
| `eidosian` | Heavy border with glow | Heavy | Glow |
| `shadowed` | Drop shadow effect | None | Shadow |
| `metallic` | Industrial look | Heavy | Emboss |
| `circuit` | Digital pattern | Single | Digital |

### Custom Styling

```python
from glyph_forge.core.style_manager import apply_style, create_custom_style

# Apply a preset style
styled_art = apply_style(ascii_art, "boxed", padding=2)

# Create a custom style
create_custom_style(
    name="my_style",
    border="rounded",
    padding=(1, 2),
    alignment="center"
)
```

## 📚 API Reference

### GlyphForgeAPI

The main API class providing unified access to all functionality.

```python
class GlyphForgeAPI:
    def generate_banner(
        self,
        text: str,
        style: Optional[str] = None,
        font: Optional[str] = None,
        width: Optional[int] = None,
        effects: Optional[List[str]] = None,
        color: bool = False,
    ) -> str: ...

    def image_to_Glyph(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        charset: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        invert: bool = False,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        dithering: bool = False,
        color_mode: str = "none",
    ) -> str: ...

    def get_available_fonts(self) -> List[str]: ...
    def get_available_styles(self) -> Dict[str, Dict[str, Any]]: ...
    def get_available_alphabets(self) -> List[str]: ...
    def save_to_file(self, glyph_art: str, file_path: str) -> bool: ...
    def preview_font(self, font: str, text: str = "Glyph Forge") -> str: ...
    def preview_style(self, style: str, text: str = "Glyph Forge") -> str: ...
```

### ImageGlyphConverter

Low-level image conversion class.

```python
class ImageGlyphConverter:
    def __init__(
        self,
        charset: str = "general",
        width: int = 100,
        height: Optional[int] = None,
        invert: bool = False,
        brightness: float = 1.0,
        contrast: float = 1.0,
        auto_scale: bool = True,
        dithering: bool = False,
        threads: int = 0,
    ): ...

    def convert(
        self,
        image_path: Union[str, Image.Image],
        output_path: Optional[str] = None,
        style: Optional[str] = None,
    ) -> str: ...

    def convert_color(
        self,
        image_path: Union[str, Image.Image],
        output_path: Optional[str] = None,
        color_mode: Union[str, ColorMode] = "ansi",
    ) -> str: ...

    def set_charset(self, charset: str, invert: bool = False) -> None: ...
    def set_image_params(...) -> None: ...
    def get_available_charsets(self) -> List[str]: ...
```

### BannerGenerator

Text banner generation engine.

```python
class BannerGenerator:
    def __init__(
        self,
        font: str = 'slant',
        width: int = 80,
        height: Optional[int] = None,
        cache_enabled: bool = True,
    ): ...

    def generate(
        self,
        text: str,
        style: str = "minimal",
        padding: Optional[Tuple[int, int]] = None,
        border: Optional[str] = None,
        alignment: str = "center",
        effects: Optional[List[str]] = None,
        color: bool = False,
    ) -> str: ...

    def available_fonts(self) -> List[str]: ...
    def preview_fonts(self, text: str, limit: int = 5) -> str: ...
    def get_metrics(self) -> Dict[str, int]: ...
    def reset_metrics(self) -> None: ...
```

## ⚡ Performance

### Benchmarks

| Operation | Input | Time | Notes |
|-----------|-------|------|-------|
| Image conversion | 800×600 | ~90ms | Level 2 optimization |
| Image conversion | 1920×1080 | ~180ms | Level 2 optimization |
| Image conversion | 4K UHD | ~420ms | Level 2 optimization |
| Image conversion | 4K UHD | ~120ms | Level 4 (parallel) |
| Banner generation | 10 chars | ~5ms | Cached |
| Banner generation | 50 chars | ~15ms | First render |

### Optimization Tips

1. **Use caching**: Banner generation caches results automatically
2. **Enable parallel processing**: Set `threads > 1` for large images
3. **Reduce output size**: Smaller `width` = faster processing
4. **Choose simpler charsets**: Fewer characters = faster mapping
5. **Disable auto-scale**: Set `auto_scale=False` if terminal size is known

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=glyph_forge --cov-report=html

# Run specific test categories
python -m pytest tests/ -m unit
python -m pytest tests/ -m integration

# Run specific test file
python -m pytest tests/test_api.py -v
```

### Test Coverage

The test suite covers:
- API functionality (21 tests)
- Banner generation (12 tests)
- Image conversion (23 tests)
- Services layer (3 tests)
- Style management (2 tests)
- Profile management (2 tests)

**Current Status**: 63 tests passing

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/Ace1928/glyph_forge.git
cd glyph_forge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run linters
black src/
isort src/
flake8 src/
mypy src/

# Build documentation
pip install -e ".[docs]"
cd docs && make html
```

### Code Style

- **Formatting**: Black with 88-character line length
- **Imports**: isort with black profile
- **Linting**: flake8 with pragmatic exceptions
- **Type Hints**: Full type annotations required (mypy strict mode)

## 🔒 Security

Security is a priority. See our [Security Policy](SECURITY.md) for:
- Vulnerability reporting procedures
- Supported versions
- Security best practices

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

## 📜 License

Glyph Forge is licensed under the MIT License.

```
Copyright (c) 2024-2025 Lloyd Handyside, Neuroforge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

*Maintained by [Lloyd Handyside](mailto:ace1928@gmail.com) and [Eidos](mailto:syntheticeidos@gmail.com) at [Neuroforge](https://neuroforge.io)*

**"Where structure embodies meaning and each character serves purpose."** ⚡

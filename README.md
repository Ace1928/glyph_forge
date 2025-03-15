# ⚡ Glyph Forge ⚡

> *"Where characters and pixels merge with structural integrity."*

glyph art transformation toolkit built on Eidosian principles. Transform images, text, and video into glyph with precision-engineered algorithms.

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)
![Status](https://img.shields.io/badge/status-beta-purple.svg)

## 📋 Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Features](#-features)
- [Usage](#-usage)
- [Configuration](#️-configuration)
- [Examples](#️-examples)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [Security](#-security)
- [License](#-license)

## 🔧 Installation

```bash
# Standard install
pip install glyph-forge

# Development version
git clone https://github.com/Ace1928/glyph_forge.git
cd glyph_forge
pip install -e .

# With extensions
pip install -e ".[all]"    # Complete toolkit
pip install -e ".[dev]"    # Development tools
pip install -e ".[docs]"   # Documentation system
```

## 🚀 Quick Start

```python
from glyph_forge import image_to_glyph

# Minimal approach
print(image_to_glyph("cat.jpg"))

# Precision control
from glyph_forge.transformers import ImageTransformer
from glyph_forge.renderers import ANSIRenderer

transformer = ImageTransformer(width=80, char_set="block")
renderer = ANSIRenderer(color_mode="truecolor")
glyph_art = transformer.transform("portrait.jpg", renderer=renderer)
print(glyph_art)  # Behold the transformation ✨
```

## ✨ Features

- **🔄 Multi-format conversion** - Images, text, videos → glyph art
- **🎭 Intelligent mapping** - Context-aware character density with edge detection
- **🌈 Color support** - True color with graceful terminal fallbacks
- **⚙️ Fine-grained control** - Customize every aspect of transformation
- **🧩 Extensible design** - Create custom transformers and renderers
- **⚡ Optimized processing** - Convert images in ~0.1s at standard settings

## 🔬 Usage

### Transform Image to glyph

```python
from glyph_forge.services import image_to_glyph

# Simple conversion
result = image_to_glyph("photo.jpg")

# Advanced configuration
result = image_to_glyph(
    "nebula.png",
    width=120,
    char_set="detailed",
    color=True,
    dither="floyd-steinberg",
    optimization_level=3
)
```

### Create Text Banners

```python
from glyph_forge.services import text_to_banner

# Quick title generation
print(text_to_banner("GLYPH FORGE"))

# Custom styling
banner = text_to_banner(
    "FORGE",
    font="slant",
    color="gradient",
    width=80,
    alignment="center"
)
print(banner)
```

### Process Video Frames

```python
from glyph_forge.services import video_to_glyph_frames
import time

# Video to glyph sequence
frames = video_to_glyph_frames(
    "clip.mp4",
    width=100,
    fps=15,
    color=True
)

# Playback or processing
for frame in frames:
    print(frame)
    time.sleep(1/15)  # 15fps playback
```

## ⚙️ Configuration

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `width` | Output width | 80 | 10-1000 |
| `height` | Output height | Auto | 10-1000 |
| `char_set` | Character mapping | "standard" | "standard", "detailed", "block", "minimal", "quantum", "eidosian", or custom string |
| `color` | Enable color | False | True/False |
| `color_mode` | Color rendering | "ansi256" | "none", "ansi16", "ansi256", "truecolor", "rgb", "web" |
| `dither` | Dither algorithm | "none" | "none", "floyd-steinberg", "jarvis", "stucki", "atkinson", "burkes", "sierra" |
| `pixel_ratio` | Width/height ratio | 0.43 | 0.2-1.0 |
| `optimization_level` | Processing mode | 2 | 0-4 (higher = faster) |

## 🖼️ Examples

Input image → glyph output:

Original        glyph Representation
🌄  →  ░░░░░░▒▒▓▓▓▓▓▓▓█████▓▓▒▒▒░░░░░
            ░░░▒▒▓▓▓██████████████▓▒░░
            ░▒▒▓▓███████████████████▓▒░
            ░▓███████████████████████▓▒

## ⚡ Performance

| Input Size | Processing Time | Output Size | Optimization |
|------------|-----------------|-------------|-------------|
| 800×600px  | 0.09s           | 100×75 chars | Level 2 |
| 1920×1080px | 0.18s          | 160×90 chars | Level 2 |
| 4K UHD     | 0.42s           | 200×112 chars | Level 2 |
| 4K UHD     | 0.12s           | 200×112 chars | Level 4 |

- *Measured on AMD Ryzen 7, 32GB RAM, Python 3.12*

## 🤝 Contributing

Glyph Forge welcomes your contributions:

1. Fork the repository
2. Create branch: `git checkout -b feature/new-capability`
3. Apply [Eidosian principles](docs/source/reference/eidosian_principles.md)
4. Commit changes: `git commit -m 'Add feature with clear purpose'`
5. Push branch: `git push origin feature/new-capability`
6. Open a pull request

Our development philosophy: every character earns its place.

## 🔒 Security

Security is a priority. See our [Security Policy](SECURITY.md) for vulnerability reporting procedures and implemented measures.

## 📜 License

Glyph Forge is licensed under the MIT License.

Copyright (c) 2024-2025 Lloyd Handyside, Neuroforge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to conditions in the LICENSE file.

*Maintained by [Lloyd Handyside](mailto:ace1928@gmail.com) and [Eidos](mailto:syntheticeidos@gmail.com) at [Neuroforge](https://neuroforge.io)*

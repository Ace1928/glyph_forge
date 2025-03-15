#!/usr/bin/env python3
# ⚡ Eidosian Setup Bridge ⚡
"""
⚡ ASCII FORGE SETUP ⚡
~~~~~~~~~~~~~~~~~~~~~

Quantum-aligned compatibility layer for ASCII Forge.

This file bridges compatibility with legacy build tools that don't 
support pyproject.toml. It delegates all configuration through setuptools
while maintaining Eidosian structural integrity.
"""
import re
from pathlib import Path
from setuptools import setup, find_packages

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔮 Version detection with Eidosian precision
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_version():
    """Extract version from package with quantum certainty."""
    version_file = Path("src") / "ascii_forge" / "__init__.py"
    with open(version_file, encoding="utf-8") as f:
        version_match = re.search(r"VERSION\s*=\s*\((\d+),\s*(\d+),\s*(\d+)\)", f.read())
        if version_match:
            return ".".join(version_match.groups())
    return "0.1.0"  # Fallback with grace

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📦 Package metadata extraction - Cross-dimensional consistency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_long_description():
    """Extract long description from README with Eidosian thoroughness."""
    readme_path = Path("README.md")
    if readme_path.exists():
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return "ASCII Forge - Hyper-optimized ASCII art converter with Eidosian principles"

def get_package_metadata():
    """Extract core package metadata with perfect alignment."""
    init_file = Path("src") / "ascii_forge" / "__init__.py"
    metadata = {
        "name": "ascii-forge",
        "version": get_version(),
        "description": "Hyper-optimized text, image and video-to-ASCII art converter with Eidosian principles",
        "author": "Lloyd Handyside",
        "author_email": "ace1928@gmail.com",
        "license": "MIT",
        "maintainer": "Neuroforge",
        "maintainer_email": "lloyd.handyside@neuroforge.io",
    }
    
    if init_file.exists():
        with open(init_file, encoding="utf-8") as f:
            content = f.read()
            # Extract metadata from __init__.py if available
            for field in ["author", "license", "email", "maintainer"]:
                pattern = fr"__(?:{field})__\s*=\s*['\"]([^'\"]+)['\"]"
                match = re.search(pattern, content)
                if match and match.group(1):
                    metadata[field] = match.group(1)
    
    return metadata

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🧩 Dependencies - Quantum entanglement matrix
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INSTALL_REQUIRES = [
    "pillow>=9.0.0",
    "numpy>=1.26.0",
    "pyfiglet>=0.8.0",
    "colorama>=0.4.6",
    "rich>=13.7.0", 
    "typer>=0.9.0",
]

DEV_REQUIRES = [
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=4.0.0",
    "mypy>=0.950",
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
]

DOCS_REQUIRES = [
    "sphinx>=8.2.3",
    "furo>=2024.8.6",
    "myst-parser>=4.0.1",
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🌀 Build system activation - Dimensional gateway
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # 📡 Extract metadata with quantum precision
    metadata = get_package_metadata()
    
    print("⚡ Activating ASCII Forge setup bridge...")
    setup(
        # Core identity
        name=metadata["name"],
        version=metadata["version"],
        description=metadata["description"],
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        
        # Authorship - Eidosian collective
        author=metadata["author"],
        author_email=metadata["author_email"],
        maintainer="Neuroforge",
        maintainer_email="lloyd.handyside@neuroforge.io",
        
        # License and platform
        license=metadata["license"],
        python_requires=">=3.12",
        
        # Package structure
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        include_package_data=True,
        
        # Entry points for CLI tools
        entry_points={
            "console_scripts": [
                "asciify=ascii_forge.cli.asciify:main",
                "bannerize=ascii_forge.cli.bannerize:main",
            ],
        },
        
        # Dependencies with quantum entanglement
        install_requires=INSTALL_REQUIRES,
        extras_require={
            "dev": DEV_REQUIRES,
            "docs": DOCS_REQUIRES,
            "all": DEV_REQUIRES + DOCS_REQUIRES,
        },
        
        # Classification for quantum indexing
        classifiers=[
            "Development Status :: 4 - Beta",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.12",
            "License :: OSI Approved :: MIT License",
            "Topic :: Artistic Software",
            "Topic :: Multimedia :: Graphics :: Graphics Conversion",
            "Topic :: Utilities",
        ],
        
        # Project URLs for multidimensional navigation
        project_urls={
            "Homepage": "https://github.com/Ace1928/ascii_forge",
            "Bug Tracker": "https://github.com/Ace1928/ascii_forge/issues",
            "Documentation": "https://ascii-forge.readthedocs.io/",
            "Organization": "https://neuroforge.io",
        },
    )
    print("✨ Setup bridge executed with Eidosian precision")
    print("🔮 Dimensional alignment complete - ASCII Forge is ready")

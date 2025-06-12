#!/usr/bin/env python3
# ⚡ Eidosian Glyph Transformation System ⚡
"""
⚡ GLYPH FORGE ⚡
~~~~~~~~~~~~~~~~

Zero-compromise Glyph art transformation toolkit with Eidosian precision.
Where structure embodies meaning and each character serves purpose.

🔥 Core capabilities:
• Character mapping with contextual intelligence
• Multi-backend rendering with adaptive output
• Precise styling with atomic customization
• Transformation pipeline with deterministic results
• Terminal-aware color with graceful degradation
"""

import os
import logging
import sys
from typing import Dict, List, Union, Callable, TypeVar, Protocol, Any, Tuple
from typing import TypedDict, Final, Literal
from pathlib import Path
from functools import lru_cache

from .services import image_to_glyph, text_to_banner, video_to_glyph_frames

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 Version and identity - The essence of our being
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VERSION: Final[Tuple[int, int, int]] = (0, 1, 0)  # Semantic versioning
__version__ = ".".join(map(str, VERSION))
__author__ = "Lloyd Handyside"
__license__ = "MIT"
__maintainer__ = "Neuroforge"
__email__ = "ace1928@gmail.com"
__maintainer_email__ = "lloyd.handyside@neuroforge.io"
__status__ = "Beta"
__copyright__ = "Copyright 2023-2024 Neuroforge"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 Self-aware logging system - Introspective feedback loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Configure root logger before anything else - structural foundation
logging.basicConfig(
    level=os.environ.get("GLYPH_FORGE_LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Package logger with namespace isolation
logger = logging.getLogger("glyph_forge")
logger.debug(f"Glyph Forge v{__version__} initializing - structural integrity check")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🏗️ Project information - The architecture of our universe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MaintainerInfo(TypedDict):
    """Maintainer contact information."""
    name: str
    email: str

class ProjectInfo(TypedDict):
    """Complete project metadata structure."""
    name: str
    description: str
    version: Tuple[int, int, int]
    author: str
    email: str
    organization: str
    org_email: str
    url: str
    license: str
    copyright: str
    maintainers: List[MaintainerInfo]
    repository: str
    status: str

# Project information - single source of truth
PROJECT: Final[ProjectInfo] = {
    "name": "Glyph Forge",
    "description": "Zero-compromise Glyph art transformation toolkit with Eidosian precision",
    "version": VERSION,
    "author": __author__,
    "email": __email__,
    "organization": "Neuroforge",
    "org_email": __maintainer_email__,
    "url": "https://github.com/Ace1928/glyph_forge",
    "license": __license__,
    "copyright": __copyright__,
    "maintainers": [
        {"name": __maintainer__, "email": __maintainer_email__},
        {"name": "Eidos", "email": "syntheticeidos@gmail.com"}
    ],
    "repository": "https://github.com/Ace1928/glyph_forge",
    "status": __status__
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🧬 Type definitions - The DNA of our system
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Type definitions with structural precision - no ambiguity tolerated
TransformerMap = Dict[str, Callable[[bytes], bytes]]
RenderOptions = Dict[str, Union[str, int, float, bool]]
GlyphMatrix = List[List[str]]
T = TypeVar('T')  # Generic output type
R = TypeVar('R')  # Return type variant

# Color mode literals for type safety
ColorMode = Literal["none", "ansi16", "ansi256", "truecolor", "rgb", "web"]

# Dithering algorithm literals
DitherAlgorithm = Literal[
    "none", "floyd-steinberg", "jarvis", "stucki", "atkinson", "burkes", "sierra"
]

class Renderer(Protocol[T]):
    """Protocol defining core rendering capabilities.
    
    A renderer transforms GlyphMatrix data into output of type T.
    This abstraction allows for multiple output formats while
    maintaining a consistent interface.
    """
    def render(self, matrix: GlyphMatrix, options: RenderOptions) -> T: ...

class Transformer(Protocol):
    """Protocol defining core transformation capabilities.
    
    Transforms input data into an GlyphMatrix representation.
    Each transformer specializes in specific input formats.
    """
    def transform(self, source: Any, **options: Any) -> GlyphMatrix: ...

class SystemCapabilities(TypedDict, total=False):
    """System environment detection results."""
    color_support: Dict[str, Any]
    terminal_size: Tuple[int, int]
    python_version: tuple
    platform: str
    glyph_forge_version: str
    has_pillow: bool
    has_numpy: bool
    has_rich: bool
    unicode_support: bool

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⚙️ Global configuration - The rules of our universe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Configuration with sane defaults and overridable parameters
DEFAULT_CONFIG: Final[Dict[str, Any]] = {
    "char_sets": {
        "standard": " .:-=+*#%@",  # Classic Glyph density gradient
        "detailed": " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",  # High-fidelity mapping
        "block": "░▒▓█",  # Block-based density representation
        "minimal": " ._|/\\#",  # Minimal set for constrained environments
        "eidosian": "⚡✧✦⚛⚘⚔⚙⚚⚜⛭⛯❄❈❉❊",  # Eidosian symbols for special projects
    },
    "color_modes": ["none", "ansi16", "ansi256", "truecolor", "rgb", "web"],
    "default_width": 80,  # Standard terminal width
    "default_height": 24,  # Standard terminal height
    "dither_algorithms": [
        "none",  # No dithering
        "floyd-steinberg",  # Classic error diffusion
        "jarvis",  # Enhanced diffusion pattern
        "stucki",  # Improved error weights
        "atkinson",  # Balanced for typography
        "burkes",  # Modified diffusion
        "sierra",  # Three-line diffusion
    ],
    "edge_detection": True,  # Preserve structural boundaries
    "structure_path": {
        "temp": Path(os.path.expanduser("~/.glyph_forge/temp")),
        "cache": Path(os.path.expanduser("~/.glyph_forge/cache")), 
        "output": Path(os.path.expanduser("~/.glyph_forge/output")),
        "resources": Path(__file__).parent / "resources",
        "config": Path(os.path.expanduser("~/.glyph_forge/config"))
    }
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📦 Core imports - The building blocks of our system
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
try:
    # Core API with clean entrypoints
    from .api import get_api, GlyphForgeAPI

    # Rendering engines for diverse output formats
    from .renderers import TextRenderer, HTMLRenderer, ANSIRenderer, SVGRenderer

    # Transformation modules for intelligent processing
    from .transformers import ImageTransformer, ColorMapper, DepthAnalyzer, EdgeDetector

    # Utility functions for seamless integration
    from .utils import setup_logger, configure, measure_performance, detect_capabilities

    # Integration services for common workflows
    from .services import text_to_banner, video_to_glyph_frames
except ImportError as e:
    # Handle partial installations with grace - no component left behind
    logger.warning(f"Module initialization incomplete: {e}")
    logger.warning("Some functionality may be unavailable - adjust expectations accordingly")
    
    # Create stub definitions for type checking
    def get_api(*args: Any, **kwargs: Any) -> Any: ...
    class GlyphForgeAPI: ...
    class TextRenderer: ...
    class HTMLRenderer: ...
    class ANSIRenderer: ...
    class SVGRenderer: ...
    class ImageTransformer: ...
    class ColorMapper: ...
    class DepthAnalyzer: ...
    class EdgeDetector: ...
    def setup_logger(*args: Any, **kwargs: Any) -> logging.Logger: return logger
    def configure(*args: Any, **kwargs: Any) -> None: ...
    def measure_performance(func: Callable[..., R]) -> Callable[..., R]: return func
    def detect_capabilities() -> Dict[str, Any]: return {}
    def image_to_glyph(*args: Any, **kwargs: Any) -> str: return ""
    def text_to_banner(*args: Any, **kwargs: Any) -> str: return ""
    def video_to_glyph_frames(*args: Any, **kwargs: Any) -> List[str]: return []

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔧 Core functions - Essential operational capabilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@lru_cache(maxsize=1)
def get_config(profile: str = None) -> Dict[str, Any]:
    """Get configuration with adaptive precision. ⚙️
    
    Configuration follows this hierarchy (highest to lowest precedence):
    1. Environment variables (GLYPH_FORGE_*)
    2. User profile selection
    3. System defaults
    
    Args:
        profile: Optional config profile name ("minimal", "standard", "eidosian")
    
    Returns:
        Dictionary with configuration parameters
        
    Example:
        >>> config = get_config("eidosian")
        >>> print(config["char_sets"]["active"])
        ⚡✧✦⚛⚘⚔⚙⚚⚜⛭⛯❄❈❉❊
    """
    # Deep copy to avoid mutating defaults
    config = {**DEFAULT_CONFIG}
    
    # Set active character set based on profile
    config["char_sets"]["active"] = config["char_sets"]["standard"]
    
    # Apply profile-specific overrides
    if profile:
        if profile == "minimal":
            config["char_sets"]["active"] = config["char_sets"]["minimal"]
            config["optimization_level"] = 1
        elif profile == "standard":
            config["char_sets"]["active"] = config["char_sets"]["standard"]
            config["optimization_level"] = 2
        elif profile == "detailed":
            config["char_sets"]["active"] = config["char_sets"]["detailed"]
            config["optimization_level"] = 3
        elif profile == "eidosian":
            config["char_sets"]["active"] = config["char_sets"]["eidosian"]
            config["optimization_level"] = 4
            config["entropy_preservation"] = True
    
    # Override with environment variables - environment trumps defaults
    prefix = "GLYPH_FORGE_"
    for key, value in config.items():
        env_key = f"{prefix}{key.upper()}"
        if env_key in os.environ:
            env_value = os.environ[env_key]
            
            # Type-aware conversion based on default value type
            if isinstance(value, bool):
                config[key] = env_value.lower() in ("true", "1", "yes", "y")
            elif isinstance(value, int):
                config[key] = int(env_value)
            elif isinstance(value, float):
                config[key] = float(env_value)
            elif isinstance(value, list):
                config[key] = env_value.split(",")
            else:
                config[key] = env_value
    
    return config

def get_project_info() -> ProjectInfo:
    """Get project information with Eidosian clarity. 📝
    
    Returns:
        Dictionary with complete project metadata
        
    Example:
        >>> info = get_project_info()
        >>> print(f"Using {info['name']} v{'.'.join(map(str, info['version']))}")
        Using Glyph Forge v0.1.0
    """
    return PROJECT

def get_system_capabilities() -> SystemCapabilities:
    """Detect system capabilities for optimal operation. 💻
    
    Identifies environment characteristics to enable
    adaptive behavior without explicit configuration.
    
    Returns:
        Dictionary with detected system capabilities
        
    Example:
        >>> caps = get_system_capabilities()
        >>> print(f"Terminal size: {caps['terminal_size'][0]}x{caps['terminal_size'][1]}")
        Terminal size: 80x24
    """
    if "detect_capabilities" not in globals():
        # Fallback if module not fully loaded
        term_size = os.get_terminal_size() if sys.stdout.isatty() else (80, 24)
        return {
            "color_support": {"level": "basic"},
            "terminal_size": term_size,
            "python_version": sys.version_info,
            "platform": sys.platform,
            "glyph_forge_version": __version__,
            "has_pillow": _has_module("PIL"),
            "has_numpy": _has_module("numpy"),
            "has_rich": _has_module("rich"),
            "unicode_support": sys.stdout.encoding and "utf" in sys.stdout.encoding.lower(),
        }
    
    # Full capability detection
    capabilities: SystemCapabilities = {
        "color_support": detect_capabilities(),
        "terminal_size": os.get_terminal_size() if sys.stdout.isatty() else (80, 24),
        "python_version": sys.version_info,
        "platform": sys.platform,
        "glyph_forge_version": __version__,
        "has_pillow": _has_module("PIL"),
        "has_numpy": _has_module("numpy"),
        "has_rich": _has_module("rich"),
        "unicode_support": sys.stdout.encoding and "utf" in sys.stdout.encoding.lower(),
    }
    return capabilities

def _has_module(module_name: str) -> bool:
    """Check if a module is available. 📦
    
    Internal utility for capability detection.
    
    Args:
        module_name: Name of the module to check
        
    Returns:
        True if module can be imported, False otherwise
    """
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🌐 Public API - The interface to our universe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__all__: List[str] = [
    # Core API
    "get_api", "GlyphForgeAPI",
    
    # Renderers
    "TextRenderer", "HTMLRenderer", "ANSIRenderer", "SVGRenderer",
    
    # Transformers
    "ImageTransformer", "ColorMapper", "DepthAnalyzer", "EdgeDetector",
    
    # Utilities
    "setup_logger", "configure", "measure_performance", "detect_capabilities",
    
    # Services
    "image_to_glyph", "text_to_banner", "video_to_glyph_frames",
    
    # Type definitions
    "TransformerMap", "RenderOptions", "GlyphMatrix", "Renderer", "Transformer",
    "SystemCapabilities", "ColorMode", "DitherAlgorithm",
    
    # Global functions
    "get_config", "get_project_info", "get_system_capabilities",
    
    # Version info
    "__version__", "__author__", "__license__", "__email__"
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔎 Runtime initialization - The system comes alive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Ensure critical paths exist - structural integrity
for path_name, path_value in DEFAULT_CONFIG["structure_path"].items():
    if path_name in ("resources", "config"):
        # These come with the package - no need to create
        continue
    
    try:
        path_value.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured path exists: {path_value}")
    except Exception as e:
        logger.warning(f"Failed to create path {path_value}: {e}")

# Perform runtime system capability detection
try:
    system_info = detect_capabilities() if "detect_capabilities" in globals() else {}
    logger.debug(f"⚡ System capabilities detected: {len(system_info)} features")
except Exception as e:
    # Handle capability detection failures gracefully
    logger.warning(f"System capability detection failed: {e}")
    system_info = {}

# Register signal handlers for graceful termination
if sys.platform != "win32":
    try:
        import signal
        def exit_handler(sig: int, frame: Any) -> None:
            """Handle termination signals with grace and precision."""
            logger.debug(f"Signal {sig} received, exiting with structural integrity")
            sys.exit(0)
            
        signal.signal(signal.SIGINT, exit_handler)
        signal.signal(signal.SIGTERM, exit_handler)
    except Exception as e:
        logger.warning(f"Signal handler registration failed: {e}")

# Log successful initialization
logger.info(f"Glyph Forge v{__version__} initialized ⚡")

# "A blank screen without Glyph art is like code without documentation—
#  technically functional but missing the context that gives it meaning." -Eidos

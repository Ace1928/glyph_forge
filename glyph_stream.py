#!/usr/bin/env python3
"""Glyph Stream: Dimensional Unicode Art Transmutation Engine.

High-performance terminal rendering system for transforming visual content into
dimensional Unicode art with adaptive quality and real-time processing.

Features:
    * Multi-algorithm edge detection with directional awareness
    * Adaptive quality scaling based on system capabilities
    * Real-time performance monitoring with concurrent processing
    * Intelligent caching and resource management
    * Cross-platform terminal rendering with Unicode support

Attributes:
    THREAD_POOL (ThreadPoolExecutor): Optimized executor for concurrent operations
    CONSOLE (Console): Enhanced terminal output interface
"""
from __future__ import annotations

# Core system functionality
import os
import sys
import re
import io
import math
import json
import time
import uuid
import shlex
import socket
import shutil
import psutil
import random
import platform
import tempfile
import threading
import traceback
import functools
import argparse
import importlib
import unicodedata
import collections
from datetime import datetime
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from functools import lru_cache
from pathlib import Path
from typing import (
    Any, Dict, List, Literal, Mapping, NamedTuple, 
    Optional, Tuple, TypedDict, TypeVar, Union
)
import cv2
import rich.cells
import colorama
import numpy
import numpy as np
import PIL as PIL_Image
from PIL import Image 
from PIL import ImageFont, ImageDraw
from PIL import ImageOps
from PIL import ImageFilter
from PIL import ImageEnhance
from PIL import ImageChops
from PIL import ImageSequence
from PIL import ImageStat
from PIL import ImageDraw2

import pyfiglet
import yt_dlp

import rich
from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.prompt import Prompt, Confirm
import rich.box
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.theme import Theme
from rich.color import Color
from rich.markup import escape
from rich.text import Text
from rich.console import Group
from rich.live import Live


         
CONSOLE = RichConsole(color_system='256')


# Set up optimized thread pool with system-aware configuration
cpu_count = os.cpu_count() or 4
THREAD_POOL = ThreadPoolExecutor(
    max_workers=min(32, cpu_count * 2),
    thread_name_prefix="glyph_worker"
)

# Define module capability flags
HAS_NUMPY = True
HAS_CV2 = True
HAS_PIL = True
HAS_RICH = True
HAS_PYFIGLET = True
HAS_PSUTIL = True
HAS_YT_DLP = True
HAS_COLORAMA = True

# Type definitions for enhanced code clarity
T = TypeVar('T')
R = TypeVar('R')
Density = float  # Range 0.0-1.0 for normalized density values
RGB = Tuple[int, int, int]  # RGB color components
Milliseconds = float  # Time in milliseconds
Seconds = float  # Time in seconds


NDArray = np.ndarray
ImageArray = Union[NDArray, Any]

# Standard color mapping with semantic names
COLOR_MAP: Mapping[str, RGB] = {
    "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
    "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
    "white": (255, 255, 255), "black": (0, 0, 0), "orange": (255, 165, 0),
    "purple": (128, 0, 128), "pink": (255, 192, 203), "gray": (128, 128, 128),
}


# ╔══════════════════════════════════════════════════════════════╗
# ║ 🧠 Core Type Definitions & Dimensional Classification        ║
# ╚══════════════════════════════════════════════════════════════╝

class EdgeDetector(Enum):
    """🔍 Edge detection algorithms with specialized dimensional sensitivities.
    
    Each algorithm offers precise advantages for specific visual contexts,
    with dynamic parameter optimization based on content characteristics
    and structural complexity detection.
    
    Attributes:
        SOBEL: Balanced gradient sensitivity with superior directional precision
        PREWITT: Enhanced noise immunity for high-contrast structural boundaries
        SCHARR: Superior diagonal detection with perfect rotational invariance
        LAPLACIAN: Omnidirectional boundary detection with zero-crossing precision
        CANNY: Maximum edge coherence through dual-threshold hysteresis
    """
    SOBEL = auto()    # ↔️↕️ Standard workhorse with balanced gradients
    PREWITT = auto()  # 🔊 Signal-to-noise optimization for complex scenes
    SCHARR = auto()   # ↗️↘️ Angular precision for diagonal structures
    LAPLACIAN = auto() # 🔄 Second-order differential for all directions
    CANNY = auto()    # ✂️ The surgical scalpel of edge detection


class GradientResult(TypedDict):
    """📊 Edge detection tensor container with complete vectorized components.
    
    Provides comprehensive gradient analysis with normalized magnitude
    and directional components for advanced dimensional rendering,
    maintaining perfect computational alignment across all output tensors.
    
    Attributes:
        magnitude: Normalized edge intensity tensor (0-255, uint8)
        gradient_x: Horizontal gradient component tensor (float32)
        gradient_y: Vertical gradient component tensor (float32)
        direction: Angular orientation tensor in radians (float32)
    """
    magnitude: np.ndarray[Any,Any]  # Edge strength map
    gradient_x: np.ndarray[Any,Any]  # Horizontal derivatives
    gradient_y: np.ndarray[Any,Any]  # Vertical derivatives
    direction: np.ndarray[Any,Any]   # Angular direction map


class QualityLevel(IntEnum):
    """🎚️ Rendering quality presets with adaptive performance scaling.
    
    Provides standardized dimensional quality tiers for real-time
    adaptation based on system capabilities, processing metrics,
    and available resources. Each tier precisely balances visual
    fidelity against computational efficiency.
    
    Attributes:
        MINIMAL: Maximum performance for resource-constrained systems (0)
        LOW: Reduced quality with optimized resource utilization (1)
        STANDARD: Balanced quality/performance for typical usage (2)
        HIGH: Enhanced detail with selective algorithm improvements (3)
        MAXIMUM: Maximum fidelity with comprehensive enhancement (4)
    """
    MINIMAL = 0  # 🐢 Speed over beauty - pure function
    LOW = 1      # 🚲 Basic visualization with core features
    STANDARD = 2 # 🚗 Balanced approach for most contexts
    HIGH = 3     # 🚅 Enhanced clarity with selective optimizations
    MAXIMUM = 4  # 🚀 Full detail with all enhancements enabled


class VideoInfo(NamedTuple):
    """🎬 Normalized video metadata container with validated fields.
    
    Maintains structurally perfect video source information with
    comprehensive validation, normalization, and type safety.
    Zero redundancy with maximum contextual integrity.
    
    Attributes:
        url: Direct stream address for network sources
        title: Content identifier with proper sanitization
        duration: Total playback seconds (None if streaming)
        format: Content format identifier (codec/container)
        width: Frame width in pixels (None if unknown)
        height: Frame height in pixels (None if unknown)
        fps: Frames per second (None if variable)
    """
    url: Optional[str] = None
    title: str = "Unknown"
    duration: Optional[int] = None
    format: str = "unknown"
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None

    @classmethod
    def from_capture(cls, capture: Any, source_name: str, stream_format: str) -> 'VideoInfo':
        """Extract validated metadata from capture device with perfect error handling.
        
        Creates normalized metadata from capture object with precision validation
        and automatic sanitization of all fields, ensuring dimensional consistency.
        
        Args:
            capture: OpenCV VideoCapture object or compatible source
            source_name: Descriptive content identifier (path/URL)
            stream_format: Format classification ('file', 'youtube', etc.)
        
        Returns:
            VideoInfo: Normalized metadata with validated fields
        """
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Validate with domain-specific constraints
            width = width if width > 0 else None
            height = height if height > 0 else None
            fps = fps if 0 < fps < 1000 else None
            duration = int(frame_count / fps) if frame_count > 0 and fps else None
            
            # Extract clean title from path/URL
            title = os.path.basename(source_name) if os.path.exists(source_name) else source_name
            if len(title) > 40:  # Truncate with ellipsis for display
                title = title[:37] + "..."
            
            return cls(title=title, format=stream_format, width=width, 
                      height=height, fps=fps, duration=duration)
        except Exception:
            # Graceful fallback with minimal valid information
            return cls(title=str(source_name), format=stream_format)


@dataclass(frozen=True)
class PerformanceStats:
    """📈 Immutable performance metrics with normalized analysis fields.
    
    Contains comprehensive performance data with statistical validity
    guarantees and perfect computational alignment. Optimized for
    real-time quality adaptation and system performance monitoring.
    
    Attributes:
        avg_render_time: Mean rendering time in milliseconds
        avg_fps: Mean frames per second across sample window
        effective_fps: True delivered framerate across entire runtime
        total_frames: Cumulative successfully processed frames
        dropped_frames: Frames abandoned due to timing constraints
        drop_ratio: Proportion of dropped frames (0.0-1.0)
        stability: Timing consistency coefficient (0.0-1.0)
    """
    avg_render_time: float  # ms per frame
    avg_fps: float          # frames/second (windowed)
    effective_fps: float    # frames/second (lifetime)
    total_frames: int       # successfully rendered
    dropped_frames: int     # timing constraint failures
    drop_ratio: float       # failure rate (0.0-1.0)
    stability: float        # variance-based consistency (1.0=perfect)


class TextStyle(Enum):
    """🔠 Text rendering style presets with precise aesthetic parameters.
    
    Defines self-contained rendering configurations with complete
    formatting specifications for diverse visual presentation needs.
    Each style maintains perfect contextual integrity with zero redundancy.
    
    Attributes:
        SIMPLE: Clean minimal presentation without decorative elements
        STYLED: Enhanced typography with borders and visual structure
        RAINBOW: Multi-color spectral gradient with visual depth
        RANDOM: Dynamic procedural styling with controlled variation
    """
    SIMPLE = auto()  # ✓ Just the essentials - pure function
    STYLED = auto()  # 🎨 Refined aesthetics with structure
    RAINBOW = auto() # 🌈 Spectral gradient with depth
    RANDOM = auto()  # 🎲 Controlled chaos - never the same twice

# ╔══════════════════════════════════════════════════════════════╗
# ║ 🧠 Hyperdimensional System Context & Capability Analysis     ║
# ╚══════════════════════════════════════════════════════════════╝
class SystemContext:
    """🌐 Eidosian system intelligence with contextual adaptation.
    
    A thread-safe dimensional perception system that maps the operational
    universe with perfect precision. Provides recursive capability detection,
    constraint analysis, and adaptive parameter optimization through
    deterministic environmental awareness.
    
    Adheres to the Eidosian principle of Structure as Control—every
    parameter precisely placed in an architecturally sound framework
    that requires no further justification.
    
    Attributes:
        attributes (Dict[str, Any]): Core environment properties with verified values
        capabilities (Dict[str, bool]): Binary capability flags with zero ambiguity
        constraints (Dict[str, Any]): Operational boundaries for perfect adaptation
    """
    _instance = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls) -> 'SystemContext':
        """⚡ Access the singleton instance with thread-safe initialization.
        
        Returns:
            SystemContext: The perfectly calibrated global instance
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self) -> None:
        """🔄 Initialize the contextual perception matrix with zero redundancy."""
        # Execute detection phases in perfect sequence
        self.attributes = self._detect_environment()
        self.capabilities = self._analyze_capabilities()
        self.constraints = self._determine_constraints()
        
        # Verify initialization success with minimal footprint
        self._verify_integrity()
    
    def _detect_environment(self) -> Dict[str, Any]:
        """🔍 Map dimensional boundaries with fail-safe detection paths.
        
        Creates a verified attribute map with guaranteed values regardless
        of system inconsistencies or access limitations.
        
        Returns:
            Dict[str, Any]: Environment properties with perfect integrity
        """
        # Terminal dimensions with graduated fallback strategy
        try:
            dims = shutil.get_terminal_size()
            terminal_width, terminal_height = dims.columns, dims.lines
        except (AttributeError, OSError):
            # Progressive fallbacks with guaranteed values
            terminal_width = int(os.environ.get('COLUMNS', 80))
            terminal_height = int(os.environ.get('LINES', 24))
            
            # Apply boundary enforcement
            terminal_width = max(40, min(terminal_width, 500))
            terminal_height = max(10, min(terminal_height, 200))
        
        # Hardware identification with precision typing
        cpu_info = {
            "count": os.cpu_count() or 2,
            "physical": max(1, (os.cpu_count() or 2) // 2),
            "arch": platform.machine(),
            "is_64bit": sys.maxsize > 2**32
        }
        
        # Runtime context with deterministic identification
        return {
            "terminal_width": terminal_width,
            "terminal_height": terminal_height,
            "interactive": sys.stdout.isatty(),
            "platform": platform.system(),
            "python_version": tuple(int(x) for x in platform.python_version().split('.')),
            "cpu_info": cpu_info,
            "hostname": platform.node(),
            "has_ipython": "IPython" in sys.modules,
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid()
        }
    
    def _analyze_capabilities(self) -> Dict[str, bool]:
        """⚙️ Extract capability matrix with zero-ambiguity classification.
        
        Deterministically maps system capabilities through progressive 
        detection layers with perfect boundary conditions and verification.
        
        Returns:
            Dict[str, bool]: Capability flags with binary precision
        """
        # Unicode detection with environment override awareness
        encoding = sys.stdout.encoding or "ascii"
        supports_unicode = ("utf" in encoding.lower() and not any(
            k in os.environ for k in ('NO_UNICODE', 'ASCII_ONLY')
        ))
        
        # Color support with platform-specific detection paths
        supports_color = (
            bool(colorama) or 
            sys.platform != 'win32' or
            "ANSICON" in os.environ or
            "WT_SESSION" in os.environ or
            os.environ.get('COLORTERM') in ('truecolor', '24bit') or
            os.environ.get('TERM', '').endswith(('color', '256color'))
        )
        
        # Network connectivity with timeout optimization
        has_network = self._check_network_connectivity(timeout=0.3)
        
        # Performance classification with multi-dimensional analysis
        perf_tier = self._calculate_performance_tier()
        
        # Compile capability matrix with zero redundancy
        return {
            "unicode": supports_unicode,
            "color": supports_color,
            "numpy": HAS_NUMPY,
            "opencv": HAS_CV2,
            "pillow": HAS_PIL,
            "rich": HAS_RICH,
            "network": has_network,
            "performance_tier": perf_tier,
            "hardware_accel": self._check_hardware_acceleration(),
            "animations": supports_color and sys.stdout.isatty(),
            "concurrent_processing": self.attributes["cpu_info"]["count"] > 2
        }
    
    def _check_network_connectivity(self, timeout: float = 0.3) -> bool:
        """🔌 Verify network accessibility with minimal latency impact.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            bool: True if network is accessible, False otherwise
        """
        # Multi-service resilience with progressive fallbacks
        for host, port in [("1.1.1.1", 53), ("8.8.8.8", 53)]:
            try:
                socket.create_connection((host, port), timeout=timeout).close()
                return True
            except (socket.error, socket.timeout):
                continue
        return False
    
    def _check_hardware_acceleration(self) -> bool:
        """🚀 Detect hardware acceleration capabilities across platforms.
        
        Tests for GPU acceleration with zero dependencies on drivers
        being actually functional—only their potential availability.
        
        Returns:
            bool: True if hardware acceleration is available
        """
        # CUDA detection via OpenCV with exception isolation
        if HAS_CV2 and hasattr(cv2, 'cuda'):
            try:
                return cv2.cuda.getCudaEnabledDeviceCount() > 0
            except Exception:
                pass
        
        # Platform-specific optimizations
        if self.attributes["platform"] == "Darwin":
            # Metal detection on macOS
            return True
        elif self.attributes["platform"] == "Linux":
            # Check for common Linux GPU indicators
            try:
                gpu_files = ['/proc/driver/nvidia/version', '/dev/dri/card0']
                return any(os.path.exists(f) for f in gpu_files)
            except Exception:
                pass
            
        return False
    
    def _calculate_performance_tier(self) -> int:
        """📊 Calculate system performance tier with dimensional accuracy.
        
        Maps hardware capabilities to standardized performance tiers
        using multi-factor analysis with perfect boundary conditions.
        
        Returns:
            int: Performance tier (0=minimal, 3=maximum)
        """
        # Base scoring from CPU capacity
        cpu_count = self.attributes["cpu_info"]["count"]
        
        # CPU tier with logarithmic scaling
        if cpu_count >= 16:
            cpu_score = 3
        elif cpu_count >= 8:
            cpu_score = 2
        elif cpu_count >= 4:
            cpu_score = 1
        else:
            cpu_score = 0
        
        # Memory analysis with exception isolation
        mem_score = 0
        if HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                mem_gb = mem.total / (1024**3)
                
                # Memory tier with geometric thresholds
                if mem_gb >= 16:
                    mem_score = 2
                elif mem_gb >= 8:
                    mem_score = 1
            except Exception:
                pass
        
        # Hardware acceleration bonus with perfect weighting
        accel_bonus = 1 if self._check_hardware_acceleration() else 0
        
        # Calculate overall tier with boundary enforcement
        base_tier = (cpu_score + mem_score + accel_bonus) // 2
        return max(0, min(base_tier, 3))
    
    def _determine_constraints(self) -> Dict[str, Any]:
        """🧩 Map operational boundaries for perfect adaptation.
        
        Calculates the precise constraints within which rendering operations
        must function optimally, with perfect terminal-awareness.
        
        Returns:
            Dict[str, Any]: Operational constraints with validated thresholds
        """
        # Extract core dimensions for constraint derivation
        term_width = self.attributes["terminal_width"]
        term_height = self.attributes["terminal_height"]
        perf_tier = self.capabilities["performance_tier"]
        
        # Art dimensions with perfect margin calculation
        art_width = term_width - (2 if term_width < 60 else 4)
        art_height = term_height - (4 if term_height < 20 else 6)
        
        # FPS thresholds with tier-specific allocation
        fps_map = {0: 5, 1: 10, 2: 15, 3: 30}
        
        # Compile constraints with zero redundancy
        return {
            "limited_width": term_width < 60,
            "limited_height": term_height < 20,
            "max_art_width": art_width,
            "max_art_height": art_height,
            "max_scale_factor": min(4, max(1, perf_tier + 1)),
            "default_fps": fps_map.get(perf_tier, 15),
            "max_parallel_tasks": min(32, max(2, cpu_count * 2)) if 'cpu_count' in locals() else 4,
            "performance_tier": perf_tier,
            "memory_threshold": 0.5 + (perf_tier * 0.1),  # 0.5-0.8 proportional to tier
        }
    
    def _verify_integrity(self) -> None:
        """🔐 Verify dimensional integrity with zero overhead.
        
        Ensures all required attributes exist with valid types,
        recovering from any detection failures with perfect resilience.
        """
        # Verify essential attributes with boundary enforcement
        required_attributes = {
            "terminal_width": lambda v: isinstance(v, int) and v > 0,
            "terminal_height": lambda v: isinstance(v, int) and v > 0,
            "platform": lambda v: isinstance(v, str) and v != "",
        }
        
        # Repair any detected inconsistencies
        for key, validator in required_attributes.items():
            if key not in self.attributes or not validator(self.attributes[key]):
                # Apply safe defaults with perfect type matching
                if key == "terminal_width":
                    self.attributes[key] = 80
                elif key == "terminal_height":
                    self.attributes[key] = 24
                elif key == "platform":
                    self.attributes[key] = platform.system() or "Unknown"
    
    @lru_cache(maxsize=8)
    def get_optimized_parameters(self, operation: str = "general") -> Dict[str, Any]:
        """⚡ Extract context-aware parameters with perfect optimization.
        
        Provides operation-specific parameter sets that maximize performance
        while maintaining visual quality within system constraints.
        
        Args:
            operation: Target operation type ("general", "video", "image", "text")
            
        Returns:
            Dict[str, Any]: Optimized parameters for current system context
        """
        # Extract core context for parameter derivation
        tier = self.capabilities["performance_tier"]
        limited = self.constraints["limited_width"]
        
        # Base parameters with progressive enhancement
        params = {
            "scale_factor": self.constraints["max_scale_factor"],
            "block_width": 4 if limited else 6 if tier >= 2 else 8,
            "block_height": 8 if limited else 6 if tier >= 2 else 8,
            "fps": self.constraints["default_fps"],
            "edge_mode": "enhanced" if tier > 0 else "simple",
            "color_mode": self.capabilities["color"],
            "animation_level": min(tier, 2),
            "max_width": self.constraints["max_art_width"],
            "max_height": self.constraints["max_art_height"],
        }
        
        # Operation-specific enhancement with perfect specialization
        if operation == "video":
            params.update({
                "buffer_frames": 2 if tier <= 1 else 4,
                "preprocessing": tier >= 2,
                "parallel_decode": tier >= 1,
                "adaptive_quality": True,
                "quality_headroom": 0.1 * (tier + 1)
            })
        elif operation == "image":
            params.update({
                "dithering": tier >= 2,
                "edge_threshold": 60 if tier == 0 else 50 if tier == 1 else 40,
                "algorithm": "sobel" if tier <= 1 else "scharr",
                "denoise": tier >= 2,
                "contrast_boost": 1.0 + (0.1 * tier)
            })
        elif operation == "text":
            params.update({
                "font_cache_size": 4 if tier == 0 else 8 if tier == 1 else 16,
                "enable_effects": tier >= 1,
                "max_width_ratio": 0.8 + (0.05 * tier),
                "alignment": "center" if not limited else "left"
            })
            
        return params
    
    def get_dimensions(self) -> Dict[str, int]:
        """📏 Get current terminal dimensions with live updates.
        
        Returns the most current terminal dimensions, potentially
        updating from the system if terminal has been resized.
        
        Returns:
            Dict[str, int]: Current width and height in characters
        """
        # Live update for interactive terminals
        if self.attributes["interactive"]:
            try:
                dims = shutil.get_terminal_size()
                self.attributes["terminal_width"] = dims.columns
                self.attributes["terminal_height"] = dims.lines
                
                # Recalculate art dimensions
                self.constraints["max_art_width"] = dims.columns - (2 if dims.columns < 60 else 4)
                self.constraints["max_art_height"] = dims.lines - (4 if dims.lines < 20 else 6)
                
            except (AttributeError, OSError):
                pass
                
        # Return current dimensions with perfect naming
        return {
            "width": self.attributes["terminal_width"],
            "height": self.attributes["terminal_height"]
        }

# Initialize system context singleton for global access
SYSTEM_CONTEXT = SystemContext.get_instance()

# ╔══════════════════════════════════════════════════════════════╗
# ║ 🌌 Hyperdimensional Environment & System Intelligence        ║
# ╚══════════════════════════════════════════════════════════════╝

# Core modules mapping with zero-redundancy capability detection

class EnvContext:
    """🧠 Eidosian environment intelligence with perfect contextual awareness.
    
    A thread-safe singleton providing exhaustive-yet-minimal system intelligence
    with parallel detection paths, optimized caching, and zero-overhead runtime
    adaptation. Each parameter serves a precise purpose in the recursive
    optimization hierarchy.
    
    Attributes:
        terminal (Dict[str, Any]): Dimensional boundaries of text reality
        runtime (Dict[str, Any]): Python execution context parameters
        hardware (Dict[str, Any]): Physical resource quantification
        network (Dict[str, bool]): Connectivity state vectors
        modules (Dict[str, bool]): Capability existence matrix
        capabilities (Dict[str, Any]): Synthesized potential surface
        constraints (Dict[str, Any]): Operational boundaries with perfect precision
    """
    _instance: Optional['EnvContext'] = None
    _lock: threading.RLock = threading.RLock()
    
    @classmethod
    def get(cls) -> 'EnvContext':
        """⚡ Access thread-safe singleton with zero-redundancy instantiation.
        
        Returns:
            EnvContext: The perfectly calibrated global context
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self) -> None:
        """🔄 Initialize multidimensional context with parallel detection paths."""
        # Terminal properties with immediate resolution—UI is our gateway
        self.terminal = self._detect_terminal()
        
        # Execute analysis tasks with perfect concurrency
        futures = {
            "runtime": THREAD_POOL.submit(self._analyze_runtime),
            "hardware": THREAD_POOL.submit(self._analyze_hardware),
            "network": THREAD_POOL.submit(self._analyze_network)
        }
        
        # Gather results with failsafe boundaries
        self.runtime = self._safely_get(futures["runtime"], {})
        self.hardware = self._safely_get(futures["hardware"], {})
        self.network = self._safely_get(futures["network"], {"connected": False})
        
        # Calculate emergent properties with recursive refinement
        self._update_capabilities()
        self._update_constraints()

    def _detect_terminal(self) -> Dict[str, Any]:
        """📏 Map terminal dimensions with graduated fallback strategy."""
        try:
            dims = shutil.get_terminal_size()
            width, height = dims.columns, dims.lines
        except (AttributeError, OSError):
            width, height = 80, 24
            
        # Capability detection with zero-ambiguity classification
        supports_unicode = (
            sys.stdout.encoding and 
            'utf' in sys.stdout.encoding.lower() and 
            not any(k in os.environ for k in ('NO_UNICODE', 'ASCII_ONLY'))
        )
        
        supports_color = (
            bool(colorama) or 
            sys.platform != 'win32' or 
            any(k in os.environ for k in ('ANSICON', 'WT_SESSION', 'FORCE_COLOR')) or
            os.environ.get('TERM_PROGRAM') in ('vscode', 'iTerm.app') or
            os.environ.get('COLORTERM') in ('truecolor', '24bit') or
            os.environ.get('TERM', '').endswith(('color', '256color'))
        )
        
        return {
            "width": width,
            "height": height,
            "interactive": sys.stdout.isatty(),
            "supports_unicode": supports_unicode,
            "supports_color": supports_color
        }
    
    def _analyze_runtime(self) -> Dict[str, Any]:
        """🔬 Extract runtime fabric with dimensional precision."""
        return {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": tuple(map(int, platform.python_version().split('.'))),
            "hostname": platform.node(),
            "has_metal": platform.system() == "Darwin",
            "is_64bit": sys.maxsize > 2**32,
            "is_debug": hasattr(sys, 'gettrace') and sys.gettrace() is not None,
            "is_frozen": getattr(sys, 'frozen', False)
        }
    
    def _analyze_hardware(self) -> Dict[str, Any]:
        """⚙️ Quantify computational substrate with zero overhead."""
        result = {
            "memory_total": 0,
            "memory_available": 0,
            "cpu_count": os.cpu_count() or 2,
            "cpu_physical": max(1, (os.cpu_count() or 2) // 2),
            "has_cuda": False,
            "numpy_optimized": False
        }
        
        # Memory analysis with perfect error isolation
        if psutil:
            try:
                mem = psutil.virtual_memory()
                result.update({
                    "memory_total": mem.total,
                    "memory_available": mem.available,
                    "cpu_count": psutil.cpu_count(logical=True) or result["cpu_count"],
                    "cpu_physical": psutil.cpu_count(logical=False) or result["cpu_physical"]
                })
            except Exception:
                pass  # Maintain resilience through graceful degradation
                
        # Acceleration detection with cross-platform awareness
        if cv2 and hasattr(cv2, 'cuda'):
            try:
                result["has_cuda"] = cv2.cuda.getCudaEnabledDeviceCount() > 0
            except Exception:
                pass
                
        # Optimization detection with dynamic pattern matching
        if numpy:
            try:
                config_info = str(numpy.show_config()) if hasattr(numpy, "show_config") else ""
                result["numpy_optimized"] = any(lib in config_info.lower() 
                                              for lib in ["mkl", "openblas", "accelerate", "blas"])
            except Exception:
                pass
                
        return result
    
    def _analyze_network(self) -> Dict[str, bool]:
        """🔌 Verify network fabric with minimal latency impact."""
        # Multi-service resilience with progressive fallbacks
        for host, port in [("1.1.1.1", 53), ("8.8.8.8", 53)]:
            try:
                socket.create_connection((host, port), timeout=0.5).close()
                return {"connected": True}
            except (socket.error, socket.timeout):
                continue
        return {"connected": False}
    
    def _safely_get(self, future: Future, default: T) -> T:
        """🛡️ Extract future results with perfect failure isolation."""
        try:
            return future.result(timeout=0.5)
        except (TimeoutError, Exception):
            return default
    
    def _update_capabilities(self) -> None:
        """🧪 Synthesize capability matrix from detected attributes."""
        perf_tier = self._calculate_performance_tier()
        
        self.capabilities = {
            "unicode": self.terminal["supports_unicode"],
            "color": self.terminal["supports_color"],
            "interactive": self.terminal["interactive"],
            "animations": self.terminal["interactive"] and self.terminal["supports_color"],
            "high_performance": perf_tier >= 2,
            "network": self.network.get("connected", False),
            "hardware_accel": self.hardware.get("has_cuda", False) or 
                            self.runtime.get("has_metal", False),
            "performance_tier": perf_tier,
        }
    
    def _update_constraints(self) -> None:
        """📊 Map operational boundaries for perfect adaptation."""
        term_width = self.terminal["width"]
        term_height = self.terminal["height"]
        perf_tier = self.capabilities["performance_tier"]
        
        self.constraints = {
            "limited_width": term_width < 60,
            "limited_height": term_height < 20,
            "max_art_width": term_width - (2 if term_width < 60 else 4),
            "max_art_height": term_height - (4 if term_height < 20 else 6),
            "max_scale_factor": min(4, max(1, perf_tier + 1)),
            "default_fps": 5 if perf_tier == 0 else 10 if perf_tier == 1 else 15,
            "max_memory_usage": min(1.0, 0.5 + (perf_tier * 0.2)),
            "parallel_tasks": max(2, min(8, self.hardware.get("cpu_count", 2)))
        }
    
    def _calculate_performance_tier(self) -> int:
        """📈 Extract computational potential with logarithmic scaling.
        
        Maps hardware capabilities to standardized performance tiers
        using multi-factor analysis with perfect boundary conditions.
        
        Returns:
            int: Performance tier (0=minimal, 3=maximum)
        """
        cpu_count = self.hardware.get("cpu_count", 2)
        memory_gb = self.hardware.get("memory_available", 0) / (1024**3) if self.hardware.get("memory_available", 0) > 0 else 2
        
        # CPU tier with logarithmic scaling
        cpu_score = 2 if cpu_count >= 16 else 1 if cpu_count >= 8 else -1 if cpu_count <= 2 else 0
        
        # Memory tier with geometric thresholds
        mem_score = 1 if memory_gb >= 16 else -1 if memory_gb <= 2 else 0
        
        # Hardware acceleration bonus with perfect weighting
        accel_bonus = 1 if self.hardware.get("has_cuda", False) or self.runtime.get("has_metal", False) else 0
        
        # Calculate overall tier with boundary enforcement
        return max(0, min(1 + cpu_score + mem_score + accel_bonus, 3))
    
    @lru_cache(maxsize=8)
    def get_optimal_params(self, operation: str = "general") -> Dict[str, Any]:
        """⚡ Extract context-aware parameters with perfect optimization.
        
        Provides operation-specific parameter sets that maximize performance
        while maintaining visual quality within system constraints.
        
        Args:
            operation: Target operation type ("general", "video", "image", "text")
            
        Returns:
            Dict[str, Any]: Optimized parameters for current system context
        """
        # Extract core context for parameter derivation
        tier = self.capabilities["performance_tier"]
        limited = self.constraints["limited_width"]
        
        # Base parameters with progressive enhancement
        params = {
            "scale_factor": self.constraints["max_scale_factor"],
            "block_width": 4 if limited else 6 if tier >= 2 else 8,
            "block_height": 8 if limited else 6 if tier >= 2 else 8,
            "fps": self.constraints["default_fps"],
            "edge_mode": "enhanced" if tier > 0 else "simple",
            "color_mode": self.capabilities["color"],
            "animation_level": min(tier, 2),
            "max_width": self.constraints["max_art_width"],
            "max_height": self.constraints["max_art_height"],
            "cache_ttl": 300 * (tier + 1)  # Efficient caching with tier-based TTL
        }
        
        # Operation-specific enhancement with perfect specialization
        if operation == "video":
            params.update({
                "buffer_frames": 2 if tier <= 1 else 4,
                "preprocessing": tier >= 2,
                "parallel_decode": tier >= 1,
                "adaptive_quality": True,
                "quality_headroom": 0.1 * (tier + 1)
            })
        elif operation == "image":
            params.update({
                "dithering": tier >= 2,
                "edge_threshold": 60 if tier == 0 else 50 if tier == 1 else 40,
                "algorithm": "sobel" if tier <= 1 else "scharr",
                "denoise": tier >= 2,
                "contrast_boost": 1.0 + (0.1 * tier)
            })
        elif operation == "text":
            params.update({
                "font_cache_size": 4 if tier == 0 else 8 if tier == 1 else 16,
                "enable_effects": tier >= 1,
                "max_width_ratio": 0.8 + (0.05 * tier),
                "alignment": "center" if not limited else "left"
            })
            
        return params

# Initialize environment with thread safety—our foundation
ENV = EnvContext.get()

# Project metadata with perfect integrity—our identity matrix
AUTHOR_INFO = {
    "name": "Lloyd Handyside",
    "email": "ace1928@gmail.com",
    "org": "Neuroforge",
    "org_email": "lloyd.handyside@neuroforge.io",
    "contributors": ["Eidos <syntheticeidos@gmail.com>", "Prismatic Architect <prism@neuroforge.io>"],
    "version": "1.0.1",
    "updated": "2025-03-16",
    "codename": "Prismatic Cipher",
    "release_stage": "stable",
    "license": "MIT",
    "repository": "github.com/Ace1928/glyph_forge",
    "documentation": "https://neuroforge.io/docs/glyph_stream",
    "support": "https://neuroforge.io/support",
    "keywords": [
        "unicode-art", "terminal-graphics", "dimensional-rendering", 
        "reality-transmutation", "prismatic-encoding", "visual-transcendence"
    ],
    "preferences": {
        "banner_style": "cosmic",
        "color_scheme": "prismatic",
        "edge_detection": "adaptive",
        "default_scale": 2,
        "show_banner_on_import": True
    }
}

# ╔══════════════════════════════════════════════════════════════╗
# ║ 🌠 Unified Dimensional Banner System                        ║
# ╚══════════════════════════════════════════════════════════════╝

class BannerEngine:
    """Dimensional banner system with adaptive rendering capabilities.
    
    Provides contextually-aware banner generation with intelligent style selection,
    caching, and terminal-adaptive layouts. Automatically adjusts to environment
    constraints while maintaining visual consistency.
    
    Attributes:
        terminal_width (int): Current terminal width in characters
        terminal_height (int): Current terminal height in lines
        supports_unicode (bool): Whether terminal supports Unicode characters
        supports_color (bool): Whether terminal supports ANSI color codes
        symbols (Dict[str, Dict[str, str]]): Symbol registry with unicode/ascii variants
    """
    _instance: Optional['BannerEngine'] = None
    _lock: threading.RLock = threading.RLock()
    _cache: Dict[str, Tuple[str, float]] = {}  # (content, creation_timestamp)
    _cache_ttl: float = 3600.0  # Cache lifetime in seconds
    _max_cache_entries: int = 16  # Bounded cache size
    
    @classmethod
    def get_instance(cls) -> 'BannerEngine':
        """Get singleton instance with thread-safe lazy initialization.
        
        Returns:
            BannerEngine: Thread-safe singleton instance
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self) -> None:
        """Initialize banner engine with environment-aware configuration."""
        # Terminal capabilities detection
        self.terminal_width: int = ENV.terminal["width"]
        self.terminal_height: int = ENV.terminal["height"]
        self.supports_unicode: bool = True
        self.supports_color: bool = True
        self.has_figlet: bool = True
        
        # Create symbol registry with unicode/ascii variants
        self.symbols: Dict[str, Dict[str, str]] = {
            "unicode": {
                # Borders and structure
                "top_left": "╔", "top_right": "╗", "bottom_left": "╚", "bottom_right": "╝",
                "horizontal": "═", "vertical": "║", "inner_h": "━", "inner_v": "┃",
                # Decorative elements
                "cosmic": "⚝", "dimension": "⟁", "star": "✧", "star_alt": "✦", 
                "star_filled": "✫", "diamond": "◈", "bullet": "•", "circle": "○",
                # Semantic groupings
                "stars": "✧✦✫", "corners": "╔╗╚╝", "lines": "═║",
                "stylized_bullet": "◆", "stylized_star": "★",
            },
            "ascii": {
                # ASCII fallbacks
                "top_left": "+", "top_right": "+", "bottom_left": "+", "bottom_right": "+",
                "horizontal": "-", "vertical": "|", "inner_h": "=", "inner_v": "|",
                "cosmic": "#", "dimension": "+", "star": "*", "star_alt": "*",
                "star_filled": "*", "diamond": "<>", "bullet": "*", "circle": "O",
                "stars": "*-*", "corners": "++++", "lines": "-|",
                "stylized_bullet": "+", "stylized_star": "*",
            }
        }
    
    def display(self, style: str = "auto", color: bool = True, 
                width: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Display dimensional banner with environment-adaptive formatting.
        
        Renders a banner using the optimal style for current terminal dimensions
        and capabilities, with intelligent caching for performance.
        
        Args:
            style: Banner style ('auto', 'full', 'compact', 'minimal', 'ascii', 'cosmic')
            color: Whether to enable ANSI colors
            width: Custom width in characters (None=auto-detect)
            metadata: Optional custom content to include in banner
        """
        banner = self.generate(
            style if style != "auto" else self._select_optimal_style(width),
            color and self.supports_color,
            width or self.terminal_width,
            metadata
        )
        
        print(banner) if not CONSOLE else CONSOLE.print(banner)
    
    def generate(self, style: str = "cosmic", color: bool = True,
                width: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate dimensional banner with specified parameters.
        
        Creates a banner with the requested style, adapting to terminal constraints
        and applying semantic colors if supported. Uses efficient caching for
        repeated generations.
        
        Args:
            style: Banner style ('full', 'compact', 'minimal', 'ascii', 'cosmic')
            color: Whether to enable ANSI colors
            width: Custom width (None=auto-detect)
            metadata: Optional custom metadata to include
            
        Returns:
            str: Fully rendered banner text
        """
        # Parameter normalization with environment awareness
        width = width or self.terminal_width
        use_color = color and self.supports_color
        use_unicode = self.supports_unicode and style != "ascii"
        
        # Progressive style adaptation for constrained environments
        if width < 40:
            style = "minimal"  # Ultra compact for tiny terminals
        elif width < 80 and style in ("full", "cosmic"):
            style = "compact"  # Reduced size for small terminals
        
        # Force ASCII for non-Unicode environments
        if not use_unicode:
            style = "ascii"
        
        # Generate cache key with all relevant parameters
        cache_key = f"{style}:{width}:{use_color}:{use_unicode}"
        if metadata:
            cache_key += f":{hash(tuple(sorted(metadata.items())))}"
            
        # Check cache with thread safety and TTL validation
        with self._lock:
            if cache_key in self._cache:
                content, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    return content
        
        # Generate banner from template with complete context
        context = self._build_context(width, use_unicode, metadata)
        banner = self._get_template(style, width).format(**context)
        
        # Apply semantic colors if requested and supported
        if use_color and not CONSOLE:  # Skip if Rich will handle coloring
            banner = self._apply_colors(banner)
        
        # Update cache with bounded size management
        with self._lock:
            if len(self._cache) >= self._max_cache_entries:
                # Remove oldest entry
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            
            # Store with current timestamp
            self._cache[cache_key] = (banner, time.time())
        
        return banner
    
    def _select_optimal_style(self, width: Optional[int] = None) -> str:
        """Select most appropriate banner style based on terminal constraints.
        
        Args:
            width: Terminal width in characters (None=use current width)
            
        Returns:
            str: Best style for current environment
        """
        width = width or self.terminal_width
        preference = AUTHOR_INFO["preferences"].get("banner_style", "cosmic")
        
        # Progressive style selection logic
        if width < 40:
            return "minimal"  # Ultra compact
        elif width < 80:
            return "compact"  # Reduced size
        elif not self.supports_unicode:
            return "ascii"    # ASCII compatible
        elif preference in ("cosmic", "full") and width >= 100:
            return preference  # User preference when space allows
        else:
            return "cosmic" if width >= 90 else "compact"
    
    def _build_context(self, 
                      width: int,
                      use_unicode: bool,
                      custom_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build complete template context dictionary with all required values.
        
        Args:
            width: Terminal width for layout calculations
            use_unicode: Whether Unicode characters are supported
            custom_metadata: Optional user-provided metadata
            
        Returns:
            Dict[str, Any]: Complete context dictionary for template formatting
        """
        # Select appropriate symbol set
        symbols = self.symbols["unicode" if use_unicode else "ascii"]
        
        # Core context with project metadata
        context = {
            # Version information
            "version": AUTHOR_INFO["version"],
            "codename": AUTHOR_INFO["codename"],
            "release": AUTHOR_INFO.get("release_stage", "stable"),
            
            # Attribution
            "maintainer": AUTHOR_INFO["name"],
            "email": AUTHOR_INFO["email"],
            "organization": AUTHOR_INFO["org"],
            "org_email": AUTHOR_INFO.get("org_email", ""),
            
            # Layout dimensions
            "horizontal_line": symbols["horizontal"] * (width - 2),
            "separator_line": symbols["inner_h"] * (width - 4),
            "content_width": width - 4,
        }
        
        # Add symbols and custom metadata
        context.update(symbols)
        if custom_metadata:
            context.update(custom_metadata)
        
        # Generate title art with figlet when available
        if self.has_figlet:
            title_width = max(20, min(width - 6, 120))
            try:
                figlet = pyfiglet.Figlet(font="standard", width=title_width)
                title_art = figlet.renderText("GLYPH STREAM").rstrip()
                
                # Center title within available space
                if use_unicode:
                    lines = []
                    for line in title_art.splitlines():
                        padding = max(0, (width - 6 - len(line)) // 2)
                        lines.append(" " * padding + line)
                    context["title_art"] = "\n".join(lines)
                else:
                    context["title_art"] = title_art
            except Exception:
                context["title_art"] = "GLYPH STREAM"  # Fallback on error
        else:
            context["title_art"] = "GLYPH STREAM"  # Simple fallback
        
        return context
    
    def _get_template(self, style: str, width: int) -> str:
        """Get appropriate banner template based on selected style and width.
        
        Args:
            style: Banner style name
            width: Terminal width for layout decisions
            
        Returns:
            str: Template string with format placeholders
        """
        templates = {
            # Minimal ultra-compact style
            "minimal": (
                "{top_left}{horizontal_line}{top_right}\n"
                "{vertical} GlyphStream v{version} {vertical}\n"
                "{bottom_left}{horizontal_line}{bottom_right}"
            ),
            
            # Compact style for medium terminals
            "compact": (
                "{top_left}{horizontal_line}{top_right}\n"
                "{vertical} {stars} GLYPH STREAM v{version} - {codename} {stars} {vertical}\n"
                "{vertical} Dimensional Unicode Transmutation Engine {vertical}\n"
                "{bottom_left}{horizontal_line}{bottom_right}"
            ),
            
            # ASCII-compatible style
            "ascii": (
                "+{horizontal_line}+\n"
                "| GLYPH STREAM v{version} |\n"
                "| Dimensional Unicode Engine |\n"
                "| Codename: {codename} |\n"
                "+{horizontal_line}+"
            ),
            
            # Cosmic style with balanced decoration
            "cosmic": (
                "{top_left}{horizontal_line}{top_right}\n"
                "{vertical} {title_art} {vertical}\n"
                "{vertical}{separator_line}{vertical}\n"
                "{vertical}  {star}{star_alt}{cosmic} PRISMATIC CIPHER v{version} {cosmic}{star_alt}{star}  {vertical}\n"
                "{vertical}  {dimension} Transform reality through visual transcendence {dimension}  {vertical}\n"
                "{bottom_left}{horizontal_line}{bottom_right}\n"
                "{cosmic}{star} ADAPTIVE UNIVERSALITY MATRIX INITIALIZED {star}{cosmic}"
            ),
            
            # Full detailed style
            "full": (
                "{top_left}{horizontal_line}{top_right}\n"
                "{vertical} {title_art} {vertical}\n"
                "{vertical}{separator_line}{vertical}\n"
                "{vertical} {dimension} Prismatic Cipher Edition v{version} {dimension} {vertical}\n"
                "{vertical}  {star} DIMENSIONAL GLYPH TRANSMUTATION ENGINE {star}  {vertical}\n"
                "{vertical}{separator_line}{vertical}\n"
                "{vertical}  {bullet} Maintainer: {maintainer} 「{email}」 {vertical}\n"
                "{vertical}  {dimension} Organization: {organization} 「{org_email}」 {vertical}\n"
                "{bottom_left}{horizontal_line}{bottom_right}"
            )
        }
        
        return templates.get(style, templates["cosmic"])
    
    def _apply_colors(self, banner: str) -> str:
        """Apply semantic ANSI color codes to banner elements.
        
        Maps specific banner elements to appropriate colors using optimized
        pattern matching for visual consistency and emphasis.
        
        Args:
            banner: Uncolored banner string
            
        Returns:
            str: Banner with ANSI color codes applied
        """
        # Color palette with semantic mapping
        colors = {
            "title": "\033[96m",     # Bright Cyan for titles
            "emphasis": "\033[95m",   # Bright Magenta for emphasis
            "action": "\033[92m",     # Bright Green for actions
            "metadata": "\033[93m",   # Bright Yellow for metadata
            "symbol": "\033[35m",     # Magenta for symbols
            "border": "\033[34m",     # Blue for borders
            "reset": "\033[0m"        # Reset formatting
        }
        
        # Pattern-replacement pairs for semantic colorization
        patterns = [
            # Title elements
            (r"(GLYPH STREAM)", f"{colors['title']}\\1{colors['reset']}"),
            (r"(PRISMATIC CIPHER)", f"{colors['emphasis']}\\1{colors['reset']}"),
            (r"(DIMENSIONAL|TRANSMUTATION|ENGINE)", f"{colors['emphasis']}\\1{colors['reset']}"),
            (r"(Transform\s+reality|INITIALIZED)", f"{colors['action']}\\1{colors['reset']}"),
            
            # Metadata elements
            (r"(v\d+\.\d+\.\d+)", f"{colors['metadata']}\\1{colors['reset']}"),
            (r"(Maintainer:.*|Organization:.*)", f"{colors['metadata']}\\1{colors['reset']}"),
            
            # Symbols and borders (with non-capturing lookbehind/lookahead)
            (r"([✧✦✫⚝⟁•◈★☆✩])", f"{colors['symbol']}\\1{colors['reset']}"),
            (r"([╔╗╚╝═║┌┐└┘━┃+\-|])", f"{colors['border']}\\1{colors['reset']}")
        ]
        
        # Apply all patterns sequentially
        result = banner
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)
            
        return result


# Initialize global banner engine singleton
BANNER = BannerEngine.get_instance()

def display_banner(style: str = "auto", color: bool = True, 
                  width: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Display dimensional banner with automatic environment adaptation.
    
    Renders a banner using optimal style selection, Unicode detection,
    and terminal-aware layout with semantic coloring.
    
    Args:
        style: Banner style ('auto', 'full', 'compact', 'minimal', 'ascii', 'cosmic')
        color: Whether to enable ANSI colors
        width: Custom width (None=auto-detect terminal width)
        metadata: Optional custom content to include in banner
    """
    BANNER.display(style, color, width, metadata)

class UnicodeRenderEngine:
    """Multidimensional Unicode rendering system with adaptive capabilities.
    
    A thread-safe, performance-optimized rendering engine providing contextual
    character selection based on terminal capabilities with intelligent fallbacks,
    efficient caching, and dimensional gradient mapping.
    
    Attributes:
        supports_unicode (bool): Whether terminal supports Unicode rendering
        supports_color (bool): Whether terminal supports ANSI color codes
        character_maps (Dict[str, Any]): Character sets for different rendering modes
    """
    _instance: Optional['UnicodeRenderEngine'] = None
    _lock: threading.RLock = threading.RLock()
    
    @classmethod
    def get_instance(cls) -> 'UnicodeRenderEngine':
        """Get thread-safe singleton instance with optimized initialization.
        
        Returns:
            UnicodeRenderEngine: Global singleton instance
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self) -> None:
        """Initialize engine with environment detection and optimized maps."""
        # Core state with persistence strategy
        self.cache_path = Path(os.path.expanduser("~/.cache/glyph_stream/character_maps.json"))
        
        # Environment capability detection
        self.supports_unicode = ENV.capabilities["unicode"]
        self.supports_color = ENV.capabilities["color"]
        
        # Memory-optimized character maps with LRU caching
        self.character_maps = {}
        self._initialize_maps()
        
        # High-performance color cache with thread safety
        self._color_cache: Dict[Tuple[int, int, int, bool], str] = {}
    
    def _initialize_maps(self) -> None:
        """Initialize character maps with optimal loading strategy."""
        try:
            if self._load_cached():
                return
        except Exception:
            pass  # Fail gracefully to generation
            
        self._generate_maps()
        # Non-blocking persistence with low priority
        THREAD_POOL.submit(self._save_maps)
    
    def _load_cached(self) -> bool:
        """Load maps from persistent cache with validation.
        
        Returns:
            bool: True if successfully loaded
        """
        if not self.cache_path.exists():
            return False
            
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                maps = json.load(f)
                # Validate essential keys
                if all(k in maps for k in ["gradient", "full_gradients", "edges"]):
                    self.character_maps = maps
                    return True
            return False
        except (json.JSONDecodeError, IOError):
            return False
    
    def _save_maps(self) -> None:
        """Persist maps to disk with error resilience."""
        try:
            self.cache_path.parent.mkdir(exist_ok=True, parents=True)
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.character_maps, f, ensure_ascii=False)
        except Exception:
            pass  # Non-critical operation
    
    def _generate_maps(self) -> None:
        """Generate optimized character maps for all dimensions."""
        self.character_maps = {
            # Core gradients with progressive detail
            "gradient": {
                "standard": "█▓▒░ ",
                "enhanced": "█▇▆▅▄▃▂▁▀ ",
                "blocks": "█▉▊▋▌▍▎▏ ",
                "braille": "⣿⣷⣯⣟⡿⢿⣻⣽⣾ ",
                "ascii": "@%#*+=-:. "
            },
            
            # Edge representation with directional accuracy
            "edges": {
                "horizontal": {"bold": "━", "standard": "─", "light": "╌", "ascii": "-"},
                "vertical": {"bold": "┃", "standard": "│", "light": "╎", "ascii": "|"},
                "diagonal_ne": {"bold": "╱", "standard": "╱", "ascii": "/"},
                "diagonal_nw": {"bold": "╲", "standard": "╲", "ascii": "\\"},
            },
            
            # Full character sets for different rendering contexts
            "full_gradients": {
                "standard": "█▓▒░■◆●◉○♦♥♠☻☺⬢⬡✿❀✣❄★☆✩✫✬ .  ",
                "minimal": "█▓▒░ ",
                "ascii_art": "@%#*+=-:. ",
                "braille": "⣿⣷⣯⣟⡿⢿⣻⣽⣾ "
            }
        }
    
    @lru_cache(maxsize=256)
    def get_ansi_color(self, r: int, g: int, b: int, bg: bool = False) -> str:
        """Generate 24-bit ANSI color sequence with optimized caching.
        
        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            bg: Whether this is a background color
            
        Returns:
            str: ANSI color escape sequence or empty string
        """
        if not self.supports_color:
            return ""
        
        # Normalize color components
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        return f"\033[{48 if bg else 38};2;{r};{g};{b}m"
    
    def get_gradient(self, density: float, mode: str = "standard") -> str:
        """Select optimal character for density with capability awareness.
        
        Args:
            density: Value between 0.0 (empty) and 1.0 (solid)
            mode: Gradient style name
            
        Returns:
            str: Character representing requested density
        """
        # Normalize density with bounds protection
        density = max(0.0, min(1.0, density))
        
        # Select appropriate gradient based on capabilities
        gradient_key = mode
        if not self.supports_unicode and mode != "ascii":
            gradient_key = "ascii_art"
        elif mode not in self.character_maps["full_gradients"]:
            gradient_key = "standard"
            
        gradient = self.character_maps["full_gradients"].get(gradient_key, "█▓▒░ ")
        
        # Optimized single-index lookup with bounds protection
        index = min(int(density * (len(gradient) - 1)), len(gradient) - 1)
        return gradient[index]
    
    def get_edge_char(self, grad_x: float, grad_y: float, 
                     strength: float = 1.0, style: str = "standard") -> str:
        """Select optimal edge character based on gradient direction.
        
        Args:
            grad_x: X gradient component
            grad_y: Y gradient component
            strength: Edge intensity (0.0-1.0)
            style: Edge style name
            
        Returns:
            str: Character representing the edge direction
        """
        # Capability adaptation and input normalization
        if not self.supports_unicode:
            style = "ascii"
        strength = max(0.0, min(1.0, strength))
        
        # Flat region detection for optimization
        if abs(grad_x) < 1e-6 and abs(grad_y) < 1e-6:
            return "·" if self.supports_unicode else "."
        
        # Dynamic style selection based on edge strength
        actual_style = style
        if style == "standard" and strength != 1.0:
            actual_style = "bold" if strength > 0.8 else "light" if strength < 0.3 else "standard"
        
        # 8-way directional mapping with angle classification
        angle = math.degrees(math.atan2(grad_y, grad_x)) % 180
        edges = self.character_maps["edges"]
        
        # Fast path selection for common angles
        if angle < 22.5 or angle >= 157.5:
            return edges["horizontal"][actual_style]
        elif 67.5 <= angle < 112.5:
            return edges["vertical"][actual_style]
        elif 22.5 <= angle < 67.5:
            return edges["diagonal_ne"][actual_style]
        else:  # 112.5 <= angle < 157.5
            return edges["diagonal_nw"][actual_style]
    
    def get_text_width(self, text: str) -> int:
        """Calculate display width of text with Unicode and ANSI awareness.
        
        Args:
            text: Input text with possible ANSI escape sequences
            
        Returns:
            int: Visual width in character cells
        """
        # Strip ANSI escape sequences
        clean_text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)
        
        width = 0
        for c in clean_text:
            # Fast path for ASCII
            if ord(c) < 127:
                width += 1
            # Handle wide characters and combining marks
            elif unicodedata.east_asian_width(c) in ('F', 'W'):
                width += 2
            elif unicodedata.category(c) not in ('Mn', 'Me', 'Cf'):
                width += 1
        return width
    
    def apply_color(self, text: str, r: int, g: int, b: int) -> str:
        """Apply color to text with automatic reset sequence.
        
        Args:
            text: Text to colorize
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            
        Returns:
            str: Colored text with reset sequence
        """
        if not self.supports_color or not text:
            return text
        return f"{self.get_ansi_color(r, g, b)}{text}{self.reset_code()}"
    
    def reset_code(self) -> str:
        """Get ANSI reset sequence if color is supported.
        
        Returns:
            str: ANSI reset code or empty string
        """
        return "\033[0m" if self.supports_color else ""
    
    def get_enhanced_gradient_chars(self) -> str:
        """Get full standard gradient character set for dimensional rendering.
        
        Returns:
            str: Gradient characters in density order
        """
        return self.character_maps["full_gradients"]["standard"]


# Initialize singleton instance with immediate availability
UNICODE_ENGINE = UnicodeRenderEngine.get_instance()

# Export API functions with semantic naming
get_ansi_color = UNICODE_ENGINE.get_ansi_color
get_edge_char = UNICODE_ENGINE.get_edge_char
get_gradient_char = UNICODE_ENGINE.get_gradient
get_enhanced_gradient_chars = UNICODE_ENGINE.get_enhanced_gradient_chars
reset_ansi = UNICODE_ENGINE.reset_code


# ╔══════════════════════════════════════════════════════════════╗
# ║ 📝 Text Processing Core                                     ║
# ╚══════════════════════════════════════════════════════════════╝

class GlyphTextEngine:
    """Dimensional text transmutation system with adaptive rendering.
    
    A thread-safe, multi-modal text processor providing dynamic font selection,
    progressive enhancement capabilities, and contextual awareness with optimized
    resource utilization and efficient caching.
    
    Attributes:
        fonts (List[str]): Available FIGlet fonts
        font_categories (Dict[str, List[str]]): Categorized font collections
    """
    _instance: Optional['GlyphTextEngine'] = None
    _lock: threading.RLock = threading.RLock()
    _CACHE_TTL: float = 3600.0  # Cache lifetime in seconds
    
    @classmethod
    def get_instance(cls) -> 'GlyphTextEngine':
        """Access the singleton instance with thread-safe initialization.
        
        Returns:
            GlyphTextEngine: The global engine instance
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self) -> None:
        """Initialize the engine with context-aware configuration."""
        # Environment detection with optimized parameters
        params = ENV.get_optimal_params("text")
        self._terminal_width = ENV.terminal["width"]
        self._supports_color = ENV.capabilities["color"]
        self._supports_unicode = ENV.capabilities["unicode"]
        self._max_cache_size = params.get("font_cache_size", 16)
        self._enable_effects = params.get("enable_effects", True)
        
        # Thread-safe caching system
        self._render_cache: Dict[str, Tuple[List[str], float]] = {}
        self._font_cache: Dict[str, pyfiglet.Figlet] = {}
        
        # Font system initialization
        self.fonts = self._detect_available_fonts()
        self.font_categories = self._categorize_fonts()
            
    def _detect_available_fonts(self) -> List[str]:
        """Detect available FIGlet fonts with invalid font filtering.
        
        Returns:
            List[str]: Available valid font names
        """
        if not pyfiglet:
            return ["standard"]
            
        try:
            all_fonts = pyfiglet.FigletFont.getFonts()
            # Filter problematic fonts that cause rendering issues
            blacklist = {"eftichess", "eftifont", "eftirobot", "eftiwall"}
            return [font for font in all_fonts if font not in blacklist]
        except Exception:
            # Fallback to core fonts that are most commonly available
            return ["standard", "slant", "small", "big", "block", "digital"]
    
    def _categorize_fonts(self) -> Dict[str, List[str]]:
        """Organize fonts into semantic categories for intelligent selection.
        
        Returns:
            Dict[str, List[str]]: Map of category names to font lists
        """
        # Semantic categorization by stylistic properties
        categories: Dict[str, List[str]] = {
            "standard": ["standard", "slant", "small", "mini"],
            "bold": ["block", "big", "chunky", "epic", "doom", "larry3d"],
            "script": ["script", "slscript", "cursive"],
            "simple": ["small", "mini", "lean", "sub-zero", "tiny"],
            "tech": ["digital", "binary", "bubble", "hex", "cyberlarge"],
            "stylized": ["gothic", "graffiti", "cosmic", "isometric"],
            "symbols": ["banner3-D", "ivrit", "weird", "starwars"],
            "decorative": ["bubble", "banner", "dotmatrix", "broadway"],
            "ascii_only": ["standard", "small", "mini", "straight", "lean"],
        }
        
        # Add 'all' category containing all fonts
        categories["all"] = self.fonts
        
        # Ensure categories only contain valid fonts
        return {k: [f for f in v if f in self.fonts] for k, v in categories.items()}
    
    @lru_cache(maxsize=16)
    def get_figlet_instance(self, font: str) -> pyfiglet.Figlet:
        """Get cached FIGlet instance with intelligent fallbacks.
        
        Args:
            font: Font name to initialize
            
        Returns:
            pyfiglet.Figlet: Configured FIGlet renderer
        """
        try:
            return pyfiglet.Figlet(font=font)
        except Exception:
            # Silent fallback to standard font on error
            return pyfiglet.Figlet(font="standard")
    
    def text_to_figlet(
        self, 
        text: str, 
        font: str = "standard",
        width: Optional[int] = None,
        color: Optional[Tuple[int, int, int]] = None,
        justify: Literal["left", "center", "right"] = "left",
        direction: Literal["auto", "left-to-right", "right-to-left"] = "auto"
    ) -> List[str]:
        """Convert text to FIGlet art with intelligent caching and fallbacks.
        
        Args:
            text: Input text to render
            font: FIGlet font name (falls back to closest match if unavailable)
            width: Maximum width (defaults to terminal width)
            color: RGB color tuple (None=no color)
            justify: Text alignment direction
            direction: Text flow direction
            
        Returns:
            List[str]: Lines of rendered FIGlet text with optional coloring
        """
        # Parameter normalization and cache key generation
        width = width or self._terminal_width
        cache_key = f"{text}:{font}:{width}:{justify}:{direction}"
        
        # Thread-safe cache check with TTL validation
        with self._lock:
            current_time = time.time()
            if cache_key in self._render_cache:
                result, timestamp = self._render_cache[cache_key]
                if current_time - timestamp < self._CACHE_TTL:
                    return self._apply_color(result.copy(), color)
            
            # Ensure font is available or find closest alternative
            if font not in self.fonts:
                font = self._find_similar_font(font)
                
            # Core rendering with cached FIGlet instances
            try:
                figlet = self.get_figlet_instance(font)
                figlet.width = width
                figlet.justify = justify
                figlet.direction = direction
                rendered = figlet.renderText(text)
                result = [line.rstrip() for line in rendered.splitlines()]
            except Exception:
                # Silent recovery with standard font
                result = [text] if not pyfiglet else self._fallback_render(text, width)
            
            # Bounded cache management
            self._cache_result(cache_key, result, current_time)
            
        # Apply colorization if requested
        return self._apply_color(result, color)
    
    def _fallback_render(self, text: str, width: int) -> List[str]:
        """Fallback rendering when normal rendering fails.
        
        Args:
            text: Text to render
            width: Maximum width
            
        Returns:
            List[str]: Simple rendered text
        """
        try:
            # Try with standard font as fallback
            figlet = pyfiglet.Figlet(font="standard", width=width)
            rendered = figlet.renderText(text)
            return [line.rstrip() for line in rendered.splitlines()]
        except Exception:
            # Ultimate fallback is just plain text
            return [text]
    
    def _cache_result(self, key: str, value: List[str], timestamp: float) -> None:
        """Thread-safe cache update with bounded size management.
        
        Args:
            key: Cache lookup key
            value: Result to cache
            timestamp: Creation timestamp for TTL calculation
        """
        with self._lock:
            # Ensure cache doesn't exceed size limits
            if len(self._render_cache) >= self._max_cache_size:
                # Remove oldest entries
                sorted_keys = sorted(self._render_cache.keys(), 
                                   key=lambda k: self._render_cache[k][1])
                for old_key in sorted_keys[:len(sorted_keys)//4 + 1]:
                    self._render_cache.pop(old_key, None)
            
            # Add new entry
            self._render_cache[key] = (value, timestamp)
    
    def _find_similar_font(self, requested_font: str) -> str:
        """Find closest matching font using optimized lookup strategy.
        
        Args:
            requested_font: Font name to match
            
        Returns:
            str: Name of closest available font
        """
        # Fast path for category matches
        if requested_font in self.font_categories:
            fonts = self.font_categories[requested_font]
            return fonts[0] if fonts else "standard"
            
        # Fast path for exact match
        if requested_font in self.fonts:
            return requested_font
        
        # Normalize for comparison
        req_lower = requested_font.lower()
        
        # Try prefix matching for intuitive user experience
        prefix_matches = [f for f in self.fonts if f.lower().startswith(req_lower)]
        if prefix_matches:
            return prefix_matches[0]
        
        # Fall back to fast substring matching
        substring_matches = [f for f in self.fonts if req_lower in f.lower()]
        if substring_matches:
            return substring_matches[0]
            
        # Final fallback to standard
        return "standard"
    
    def _apply_color(self, lines: List[str], color: Optional[Tuple[int, int, int]]) -> List[str]:
        """Apply ANSI color to text with environment-aware capability detection.
        
        Args:
            lines: Text lines to colorize
            color: RGB color tuple (None=no color)
            
        Returns:
            List[str]: Colorized text lines
        """
        if not color or not self._supports_color:
            return lines
            
        r, g, b = color
        return [UNICODE_ENGINE.apply_color(line, r, g, b) if line else "" for line in lines]
    
    def render_multiline_art(
        self, 
        text: str,
        font: str = "standard",
        color: Optional[Tuple[int, int, int]] = None,
        width: Optional[int] = None,
        align: Literal["left", "center", "right"] = "left"
    ) -> List[str]:
        """Render multi-line text with consistent formatting per line.
        
        Args:
            text: Multi-line text to render
            font: FIGlet font name
            color: RGB color tuple
            width: Maximum width
            align: Text alignment
            
        Returns:
            List[str]: Combined rendered output with separators
        """
        lines = text.splitlines()
        if not lines:
            return []
            
        # Process each line with optimized batching
        result: List[str] = []
        for line in lines:
            if not line.strip():
                result.append("")
                continue
                
            rendered = self.text_to_figlet(
                line, font=font, width=width, color=color, justify=align
            )
            result.extend(rendered)
            result.append("")  # Line separator
            
        # Remove trailing empty line for cleaner output
        if result and not result[-1]:
            result.pop()
            
        return result
    
    def get_font_preview(
        self, 
        text: str = "abc ABC", 
        category: Optional[str] = None,
        max_fonts: int = 20
    ) -> Dict[str, List[str]]:
        """Generate font preview with sample text and category filtering.
        
        Args:
            text: Sample text to render
            category: Font category to filter by (None=all)
            max_fonts: Maximum number of fonts to preview
            
        Returns:
            Dict[str, List[str]]: Map of font names to rendered previews
        """
        # Select fonts based on category with optimized slicing
        fonts_to_preview = (self.font_categories.get(category, []) 
                          if category else self.fonts[:max_fonts])
        
        # Bounded selection for performance
        if len(fonts_to_preview) > max_fonts:
            fonts_to_preview = fonts_to_preview[:max_fonts]
        
        # Generate previews with parallel processing for larger sets
        if len(fonts_to_preview) > 5 and ENV.constraints.get("parallel_tasks", 2) > 1:
            previews = {}
            with ThreadPoolExecutor(max_workers=min(8, ENV.constraints.get("parallel_tasks", 2))) as executor:
                futures = {font: executor.submit(
                    self.text_to_figlet, text, font=font
                ) for font in fonts_to_preview}
                
                for font, future in futures.items():
                    try:
                        previews[font] = future.result()
                    except Exception:
                        continue
            return previews
        else:
            # Sequential generation for smaller sets
            return {font: self.text_to_figlet(text, font=font) 
                   for font in fonts_to_preview}
    
    def show_font_gallery(self, text: str = "Test", category: Optional[str] = None) -> None:
        """Display interactive font gallery with rich formatting.
        
        Args:
            text: Sample text to render
            category: Font category to display
        """
        # Generate previews with category filtering
        previews = self.get_font_preview(text, category, max_fonts=30)
        
        if CONSOLE:
            # Rich display with visual styling
            title = f"🔠 Font Gallery: {category or 'All'} ({len(previews)} fonts)"
            CONSOLE.print(Panel(title, style="cyan bold"))
            
            for font_name, lines in previews.items():
                font_panel = Panel("\n".join(lines), title=font_name, border_style="blue")
                CONSOLE.print(font_panel)
        else:
            # Fallback plain text display
            print(f"=== Font Gallery: {category or 'All'} ({len(previews)} fonts) ===")
            
            for font_name, lines in previews.items():
                print(f"\n--- {font_name} ---")
                print("\n".join(lines))
                print("\n" + "-" * 40)
                
    def get_random_font(self, category: Optional[str] = None) -> str:
        """Get random font name with optional category filtering.
        
        Args:
            category: Font category to select from (None=all fonts)
            
        Returns:
            str: Random font name from the selected category
        """
        # Select font list based on category with fallback
        fonts = (self.font_categories.get(category, []) 
                if category in self.font_categories else self.fonts)
        
        # Select with pseudorandom quality
        if not fonts:
            return "standard"
            
        import random
        return random.choice(fonts)
    
    def get_font_categories(self) -> List[str]:
        """Get available font categories for selection.
        
        Returns:
            List[str]: Available font category names
        """
        return list(self.font_categories.keys())
    
    def get_font_information(self, font: str) -> Dict[str, Any]:
        """Get detailed information about a specific font with sample rendering.
        
        Args:
            font: Font name to query
            
        Returns:
            Dict[str, Any]: Font metadata including name, categories and sample
        """
        # Handle font not found with graceful fallback
        if font not in self.fonts:
            closest = self._find_similar_font(font)
            return {
                "name": closest,
                "found": False,
                "similar_to": font,
                "categories": [cat for cat, fonts in self.font_categories.items() 
                              if closest in fonts and cat != "all"],
                "sample": self.text_to_figlet("Sample", font=closest)
            }
        
        # Return comprehensive information for available font
        return {
            "name": font,
            "found": True,
            "categories": [cat for cat, fonts in self.font_categories.items() 
                          if font in fonts and cat != "all"],
            "sample": self.text_to_figlet("Sample", font=font),
            "available_sizes": ["small", "standard", "large"] 
                               if self._enable_effects else ["standard"]
        }


# Initialize the text engine with singleton access pattern
TEXT_ENGINE = GlyphTextEngine.get_instance()



# ╔══════════════════════════════════════════════════════════════╗
# ║ 🎭 Text Transformation Functions                            ║
# ╚══════════════════════════════════════════════════════════════╝

def resolve_color(color: Optional[Union[str, Tuple[int, int, int]]]) -> Optional[Tuple[int, int, int]]:
    """Convert color name or RGB tuple to normalized RGB tuple.
    
    Args:
        color: Color name string or RGB tuple
        
    Returns:
        Optional[Tuple[int, int, int]]: Normalized RGB tuple or None
    """
    if color is None:
        return None
        
    if isinstance(color, tuple) and len(color) == 3:
        return (
            max(0, min(255, color[0])),
            max(0, min(255, color[1])),
            max(0, min(255, color[2]))
        )
        
    if color:
        # Try to get from the standard color map
        color_lower = color.lower()
        if color_lower in COLOR_MAP:
            return COLOR_MAP[color_lower]
            
    # Default to white if color name not found
    return COLOR_MAP.get("white")

def text_to_art(
    text: str,
    font: str = "standard",
    color: Optional[Union[str, Tuple[int, int, int]]] = None,
    width: Optional[int] = None,
    align: Literal["left", "center", "right"] = "left",
    save_path: Optional[str] = None,
    collection_path: str = "/home/lloyd/repos/glyph_forge/banner_text.md",
    save_multiple_fonts: bool = True,
    font_count: int = 20
) -> List[str]:
    """🔠 Convert text to ASCII/FIGlet art with dimensional persistence.

    Args:
        text: Text to transmute
        font: FIGlet font name or category
        color: RGB color tuple or color name
        width: Maximum width in characters (None for terminal width)
        align: Text alignment direction
        save_path: Optional file path to save rendered text
        collection_path: Path to banner collection file
        save_multiple_fonts: Generate variants across multiple fonts
        font_count: Number of font variants to generate when save_multiple_fonts=True

    Returns:
        List[str]: Lines of rendered text art
    """
    # Render primary text with specified font
    rgb_color = resolve_color(color)
    result = TEXT_ENGINE.text_to_figlet(
        text=text, font=font, width=width, color=rgb_color, justify=align
    )
    
    # Save primary result if specified
    if save_path:
        save_text_art_to_file(text, result, save_path)
    
    # Always append to collection with multiple fonts
    if collection_path:
        save_to_banner_collection(text, collection_path, save_multiple_fonts, font_count)
        
    return result


def render_styled_text(
    text: str,
    font: str = "standard",
    color: Optional[Union[str, Tuple[int, int, int]]] = None,
    width: Optional[int] = None,
    align: Literal["left", "center", "right"] = "center",
    add_border: bool = False,
    padding: int = 0,
    style: Literal["single", "double", "rounded", "bold"] = "single",
    save_path: Optional[str] = None,
    collection_path: str = "/home/lloyd/repos/glyph_forge/banner_text.md"
) -> List[str]:
    """🎨 Render text with dimensional styling and persistent archival.

    Creates fully styled text with controlled parameters for color,
    borders and padding with intelligent parameter normalization.

    Args:
        text: Text to render
        font: FIGlet font name
        color: Color name or RGB tuple
        width: Maximum width
        align: Text alignment
        add_border: Add decorative border
        padding: Padding around text (0-10)
        style: Border style when add_border is True
        save_path: Optional file path to save rendered text
        collection_path: Path to banner collection file

    Returns:
        List[str]: Lines of styled text art
    """
    rgb_color = resolve_color(color)
    
    # Generate base text art
    lines = TEXT_ENGINE.text_to_figlet(
        text=text, font=font, width=width, color=rgb_color, justify=align
    )
    
    # Apply padding with bounds enforcement
    padding = max(0, min(10, padding))
    if padding > 0:
        padded_lines = [""] * padding  # Top padding
        padded_lines.extend(f"{' ' * padding}{line}{' ' * padding}" for line in lines)
        padded_lines.extend([""] * padding)  # Bottom padding
        lines = padded_lines
    
    # Apply border if requested
    result = add_unicode_border(lines, rgb_color, style) if add_border else lines
    
    # Save to file if specified
    if save_path:
        save_text_art_to_file(text, result, save_path)
    
    # Always collect to banner archive
    if collection_path:
        save_to_banner_collection(text, collection_path, True, font_count=20)
        
    return result


def add_unicode_border(
    lines: List[str], 
    color: Optional[Tuple[int, int, int]] = None,
    style: Literal["single", "double", "rounded", "bold"] = "single"
) -> List[str]:
    """🔲 Add decorative Unicode border around text with prismatic styling.

    Args:
        lines: Text lines to frame
        color: Border RGB color
        style: Border style preset

    Returns:
        List[str]: Text with border applied
    """
    if not lines:
        return []
    
    # Calculate maximum visible line width
    width = max((UNICODE_ENGINE.get_text_width(line) for line in lines), default=0)
    
    # Border character sets by style
    borders = {
        "single": ("╔", "╗", "╚", "╝", "═", "║"),
        "double": ("╔", "╗", "╚", "╝", "═", "║"),
        "rounded": ("╭", "╮", "╰", "╯", "─", "│"),
        "bold": ("┏", "┓", "┗", "┛", "━", "┃"),
    }
    
    # Select correct border set with fallback
    tl, tr, bl, br, h, v = borders.get(style, borders["single"])
    
    # Create framed lines with color application
    result = []
    
    # Top border
    top_border = f"{tl}{h * width}{tr}"
    result.append(UNICODE_ENGINE.apply_color(top_border, *color) if color else top_border)
    
    # Content lines with side borders
    for line in lines:
        # Calculate padding to align right border
        line_width = UNICODE_ENGINE.get_text_width(line)
        padding = " " * max(0, width - line_width)
        
        # Create framed line
        if color:
            framed_line = f"{UNICODE_ENGINE.apply_color(v, *color)}{line}{padding}{UNICODE_ENGINE.apply_color(v, *color)}"
        else:
            framed_line = f"{v}{line}{padding}{v}"
            
        result.append(framed_line)
    
    # Bottom border
    bottom_border = f"{bl}{h * width}{br}"
    result.append(UNICODE_ENGINE.apply_color(bottom_border, *color) if color else bottom_border)
    
    return result


def generate_text_art(
    text: str,
    mode: Union[str, TextStyle] = TextStyle.STYLED,
    font: Optional[str] = None,
    color: Optional[Union[str, Tuple[int, int, int]]] = None,
    save_path: Optional[str] = None,
    collection_path: str = "/home/lloyd/repos/glyph_forge/banner_text.md"
) -> List[str]:
    """✨ Generate dimensional text art with stylistic transmutation.

    Creates text art with preconfigured style combinations using both
    direct and enum-based mode selection for flexible usage patterns.

    Args:
        text: Text to render
        mode: Rendering style preset (string or TextStyle)
        font: FIGlet font or category (None for mode-specific default)
        color: Text color name or RGB values
        save_path: Optional file path for rendered output
        collection_path: Path to banner collection archives

    Returns:
        List[str]: Rendered text art
    """
    # Convert string mode to enum if needed
    if isinstance(mode, str):
        try:
            mode = TextStyle[mode.upper()]
        except KeyError:
            mode = TextStyle.STYLED
    
    # Apply mode-specific parameters
    if mode == TextStyle.SIMPLE:
        result = render_styled_text(text, font or "standard", None, 
                                   add_border=False, collection_path=None)
        
    elif mode == TextStyle.STYLED:
        result = render_styled_text(
            text, font or "slant", color or "cyan", 
            add_border=True, padding=1, collection_path=None
        )
        
    elif mode == TextStyle.RAINBOW:
        # Specialized rainbow rendering with optimized approach
        selected_font = font or TEXT_ENGINE.get_random_font("bold")
        rendered = TEXT_ENGINE.text_to_figlet(text, font=selected_font)
        
        rainbow_lines = []
        for line_idx, line in enumerate(rendered):
            if not line.strip():
                rainbow_lines.append("")
                continue
                
            hue_shift = line_idx * 15  # Color variety between lines
            
            # Generate the rainbow-colored line
            chars = []
            for i, char in enumerate(line):
                if char.strip():
                    hue = (i * 10 + hue_shift) % 360
                    r, g, b = hsv_to_rgb(hue/360, 1.0, 1.0)
                    chars.append(UNICODE_ENGINE.apply_color(char, r, g, b))
                else:
                    chars.append(char)
                    
            rainbow_lines.append("".join(chars))
        
        result = add_unicode_border(rainbow_lines, (255, 255, 255), "rounded")
        
    elif mode == TextStyle.RANDOM:
        # Generate randomized styling parameters
        selected_font = font or TEXT_ENGINE.get_random_font()
        border_style = random.choice(["single", "rounded", "bold"])
        border = bool(random.getrandbits(1))  # 50% chance of border
        
        # Random vibrant color if not specified
        if color is None:
            # Generate vibrant colors (avoid dark/muted values)
            r, g, b = hsv_to_rgb(random.random(), 0.8, 1.0)  
            selected_color = (r, g, b)
        else:
            selected_color = color
        
        result = render_styled_text(
            text, selected_font, selected_color, 
            add_border=border, padding=random.choice([0, 1, 2]),
            style=border_style, collection_path=None
        )
    else:
        # Fallback for any unexpected case
        result = text_to_art(text, font or "standard", collection_path=None)
    
    # Save result if specified
    if save_path:
        save_text_art_to_file(text, result, save_path)
        
    # Save to collection
    if collection_path:
        save_to_banner_collection(text, collection_path, True, 20)
    
    return result


def show_all_fonts(text: str = "Sample", category: Optional[str] = None) -> None:
    """Display interactive gallery of available fonts with categorization.

    Args:
        text: Sample text to render
        category: Font category filter (None to show all categories)
    """
    TEXT_ENGINE.show_font_gallery(text, category)

def save_text_art_to_file(text: str, art_lines: List[str], file_path: str) -> None:
    """💾 Save text banner to file with proper formatting.
    
    Args:
        text: Original text content
        art_lines: Rendered ASCII art lines
        file_path: Path to save the file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Create banner content
    sanitized_text = re.sub(r'[^\w\s-]', '', text).strip()
    header = f"# {sanitized_text} Banner"
    
    # Format content with proper markdown structure
    content = [
        header,
        "",
        "```ascii",
        *[line.replace('\033', '') if '\033' in line else line for line in art_lines],
        "```",
        ""
    ]
    
    # Write to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))

def save_to_banner_collection(
    text: str, 
    collection_path: str = "/home/lloyd/repos/glyph_forge/text_banners.md", 
    multiple_fonts: bool = True,
    font_count: int = 20
) -> None:
    """🗄️ Archive banner in dimensional collection with font variants.
    
    Archives the text banner in a shared collection file with proper
    markdown formatting. When multiple_fonts=True, generates variants
    across different font styles for comprehensive collection.
    
    Args:
        text: Text to render
        collection_path: Path to collection file
        multiple_fonts: Generate variants with multiple fonts
        font_count: Number of fonts to use when multiple_fonts=True
    """
    # Ensure file exists (create with header if needed)
    if not os.path.exists(collection_path):
        with open(collection_path, 'w', encoding='utf-8') as f:
            f.write("# 🔠 Text Banner Collection\n\n")
    
    # Read current file to avoid duplicates
    try:
        with open(collection_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()
    except Exception:
        existing_content = ""
    
    # Sanitize text for header usage
    sanitized_text = re.sub(r'[^\w\s-]', '', text).strip()
    header = f"# {sanitized_text} Banner"
    
    # Skip if this exact header already exists
    if header in existing_content:
        return
        
    new_content = []
    
    # Generate variants with multiple fonts
    if multiple_fonts:
        # Get a selection of fonts
        if font_count > len(TEXT_ENGINE.fonts):
            font_count = len(TEXT_ENGINE.fonts)
            
        # Select fonts from different categories
        fonts_to_use = []
        for cat in ["standard", "bold", "script", "simple", "tech", "decorative"]:
            cat_fonts = TEXT_ENGINE.font_categories.get(cat, [])
            if cat_fonts:
                fonts_to_use.append(cat_fonts[0])
                
        # Fill remaining slots with random selection
        remaining = font_count - len(fonts_to_use)
        if remaining > 0:
            random_fonts = random.sample([f for f in TEXT_ENGINE.fonts 
                                       if f not in fonts_to_use], 
                                       min(remaining, len(TEXT_ENGINE.fonts)))
            fonts_to_use.extend(random_fonts)
            
        # Generate and append each font variant
        for font in fonts_to_use:
            font_header = f"## {sanitized_text} - {font}"
            
            # Skip if this exact font header already exists
            if font_header in existing_content:
                continue
                
            try:
                # Render ASCII art with current font
                art = TEXT_ENGINE.text_to_figlet(text, font=font)
                
                # Format with proper markdown and append
                new_content.extend([
                    font_header,
                    "",
                    "```ascii",
                    *[line for line in art],
                    "```",
                    ""  # Empty line after block
                ])
            except Exception:
                continue  # Skip failed fonts
    else:
        # Just use standard font
        art = TEXT_ENGINE.text_to_figlet(text, font="standard")
        new_content.extend([
            header,
            "",
            "```ascii",
            *[line for line in art],
            "```",
            ""  # Empty line after block
        ])
    
    # Append to file
    if new_content:
        with open(collection_path, 'a', encoding='utf-8') as f:
            f.write('\n' + '\n'.join(new_content))
def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """Convert HSV color to RGB components with optimized algorithm.

    Args:
        h: Hue (0.0-1.0)
        s: Saturation (0.0-1.0)
        v: Value (0.0-1.0)

    Returns:
        Tuple[int, int, int]: RGB components (0-255)
    """
    # Normalize inputs to valid ranges
    h = max(0.0, min(1.0, h))
    s = max(0.0, min(1.0, s))
    v = max(0.0, min(1.0, v))
    
    # Fast path for grayscale
    if s <= 0.0:
        v_byte = int(v * 255)
        return (v_byte, v_byte, v_byte)
    
    h *= 6.0
    i = int(h)
    f = h - i
    
    # Pre-compute common values
    p, q, t = v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f))
    
    # Mapping based on hue sector with lookup table for optimal performance
    rgb = [
        (v, t, p),  # i=0
        (q, v, p),  # i=1
        (p, v, t),  # i=2
        (p, q, v),  # i=3
        (t, p, v),  # i=4
        (v, p, q),  # i=5
    ][i % 6]
    
    # Convert to byte values
    return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


# Font visualization and discovery functions 
@lru_cache(maxsize=4)
def show_all_fonts(text: str = "Sample", category: Optional[str] = None) -> None:
    """Display interactive gallery of available fonts with categorization.

    Args:
        text: Sample text to render
        category: Font category filter (None to show all categories)
    """
    TEXT_ENGINE.show_font_gallery(text, category)


def list_font_categories() -> List[str]:
    """List available font categories with rich formatting when available.

    Returns:
        List[str]: Available font category names
    """
    categories = TEXT_ENGINE.get_font_categories()
    
    descriptions = {
        "standard": "Common readable fonts",
        "bold": "Heavy weight striking fonts",
        "script": "Flowing cursive-style fonts",
        "simple": "Minimal, space-efficient fonts",
        "tech": "Technology and computer themed",
        "stylized": "Highly decorative unique fonts",
        "symbols": "Special character based fonts",
        "decorative": "Ornamental display fonts",
        "ascii_only": "Compatible with limited terminals",
        "all": "Complete font collection"
    }
    
    table = Table(title="🔠 Font Categories")
    table.add_column("Category", style="cyan bold")
    table.add_column("Description", style="green")
    table.add_column("Count", style="magenta")
        
    for cat in sorted(categories):
        fonts = TEXT_ENGINE.font_categories.get(cat, [])
        table.add_row(cat, descriptions.get(cat, ""), f"{len(fonts)} fonts")
            
    CONSOLE.print(table)
    return categories

class ImageProcessor:
    """🎭 Prismatic Tensor Manipulation Engine with Dimensional Awareness
    
    A zero-overhead, thread-safe tensor orchestration system providing molecular-level 
    control over visual transmutation. Implements vectorized algorithms with
    perfect caching and context-adaptive parameter selection across
    the entire dimensional spectrum.
    
    Attributes:
        system_tier (int): Computational potential classification (0-3)
        max_dim (int): Upper dimensional boundary constraint
    """
    _instance: Optional['ImageProcessor'] = None
    _lock: threading.RLock = threading.RLock()
    
    @classmethod
    def get_instance(cls) -> 'ImageProcessor':
        """⚡ Access singleton transmutation core with zero overhead.
        
        Returns:
            ImageProcessor: The perfectly calibrated global processor
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self) -> None:
        """🧠 Initialize tensor manipulation core with self-optimizing kernels."""
        # Zero-overhead state initialization
        self._cache: Dict[str, Any] = {}
        self._kernels = self._build_kernels()
        
        # System-calibrated processing parameters with constraint awareness
        self.system_tier = ENV.capabilities.get("performance_tier", 1)
        self.max_dim = 8192 if self.system_tier >= 2 else 4096
        
        # Algorithm dispatcher with perfect function mapping
        self._algo_map = {
            EdgeDetector.SOBEL: self._detect_sobel,      # ↔️↕️ Balanced sensitivity
            EdgeDetector.PREWITT: self._detect_prewitt,  # 🔊 Signal-noise optimization
            EdgeDetector.SCHARR: self._detect_scharr,    # ↗️↘️ Angular precision
            EdgeDetector.LAPLACIAN: self._detect_laplacian, # 🔄 Omnidirectional
            EdgeDetector.CANNY: self._detect_canny       # ✂️ Maximum coherence
        }
    
    def _build_kernels(self) -> Dict[str, np.ndarray]:
        """🧩 Generate optimized convolution kernels with perfect symmetry.
        
        Creates mathematically precise transformation matrices for directional
        sensitivity and perceptual enhancement with zero computational waste.
        
        Returns:
            Dict[str, np.ndarray]: Named convolution kernels with optimal weights
        """
        return {
            # Edge detection kernels with perfect directionality
            "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
            "prewitt_x": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            "prewitt_y": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
            "scharr_x": np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]),
            "scharr_y": np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]]),
            "laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
            
            # Enhancement kernels with perceptual optimization
            "gaussian": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
            "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        }
    
    def supersample_image(self, image: Image.Image, scale_factor: int) -> Image.Image:
        """🔍 Upscale visual matrix with tier-adaptive resampling.
        
        Recursively enhances detail through dimensional expansion with
        context-aware algorithm selection and boundary protection.
        
        Args:
            image: Source dimensional matrix
            scale_factor: Detail enhancement coefficient (1-4)
            
        Returns:
            Image.Image: Enhanced dimensional matrix
        """
        # Parameter normalization with boundary enforcement
        scale = max(1, min(4, int(scale_factor)))
        
        # Dimensional safety check with predictive scaling
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        
        if new_width > self.max_dim or new_height > self.max_dim:
            # Adaptive downscaling to prevent dimensional overflow
            scale_factor = min(self.max_dim / image.width, self.max_dim / image.height)
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
        
        # Algorithm selection with perfect tier mapping
        resampling = Image.LANCZOS if self.system_tier >= 2 else Image.BILINEAR
        return image.resize((new_width, new_height), resampling)
    
    def rgb_to_gray(self, image_array: np.ndarray) -> np.ndarray:
        """🌓 Generate perceptual luminance map with ITU-R BT.601 precision.
        
        Transforms RGB chromatic space into perceptually-weighted grayscale
        using internationally standardized coefficients for human visual accuracy.
        
        Args:
            image_array: RGB tensor (H×W×3) or grayscale (H×W)
            
        Returns:
            np.ndarray: Perceptual luminance tensor (H×W)
        """
        # Fast path for already grayscale images
        if len(image_array.shape) == 2:
            return image_array
        
        # Vectorized dot product with perceptual coefficients (ITU-R BT.601)
        return np.dot(image_array[..., :3], [0.2126, 0.7152, 0.0722]).astype(np.uint8)
    
    def enhance_image(self, 
                     image_array: np.ndarray, 
                     contrast: float = 1.0,
                     brightness: float = 0.0,
                     denoise: bool = False) -> np.ndarray:
        """✨ Apply prismatic enhancement with vectorized precision.
        
        Transforms visual perception through multi-parameter enhancement
        with perfect boundary preservation and noise reduction.
        
        Args:
            image_array: Source visual tensor
            contrast: Perceptual range multiplier (0.5-2.0, 1.0=neutral)
            brightness: Luminance shift (-128 to +128)
            denoise: Apply Gaussian noise reduction
            
        Returns:
            np.ndarray: Enhanced visual tensor
        """
        # Fast path for identity operations with zero overhead
        if contrast == 1.0 and brightness == 0 and not denoise:
            return image_array
            
        # Parameter normalization with boundary enforcement
        contrast = np.clip(contrast, 0.5, 2.0)
        brightness = np.clip(brightness, -128, 128)
        
        # Create working copy with floating-point precision
        result = image_array.astype(np.float32)
        
        # Apply contrast and brightness in one vectorized operation
        if contrast != 1.0 or brightness != 0:
            # Use contrast formula with perfect perceptual scaling:
            # f(p) = (259(c+255))/(255(259-c)) * (p-128) + 128
            f = 259 * (contrast * 255 + 255) / (255 * (259 - contrast * 255))
            result = np.clip(f * (result - 128) + 128 + brightness, 0, 255)
            
        # Apply denoising with optimized kernel convolution
        if denoise:
            kernel = self._kernels["gaussian"]
            
            if len(result.shape) == 2:
                # Single-channel fast path for grayscale
                result = self._convolve(result, kernel)
            else:
                # Multi-channel processing with memory optimization
                for c in range(result.shape[2]):
                    result[:,:,c] = self._convolve(result[:,:,c], kernel)
                    
        return result.astype(np.uint8)
    
    def detect_edges(self, 
                    gray_array: np.ndarray, 
                    algorithm: Union[str, EdgeDetector] = EdgeDetector.SOBEL,
                    threshold: Optional[int] = None) -> GradientResult:
        """🔪 Extract edge tensors with adaptive multi-algorithm precision.
        
        Maps visual discontinuities through optimized kernel operations
        with directional sensitivity and contextual thresholding.
        
        Args:
            gray_array: Luminance tensor (H×W)
            algorithm: Edge detection algorithm specifier
            threshold: Edge sensitivity threshold (None=auto-calibrate)
            
        Returns:
            GradientResult: Complete edge mapping with directional components
        """
        # Normalize algorithm type with resilient parsing
        if isinstance(algorithm, str):
            algorithm = self._parse_algorithm(algorithm)
            
        # Auto-threshold based on image statistics with perceptual scaling
        if threshold is None:
            mean_value = np.mean(gray_array)
            threshold = int(40 + (mean_value / 5))  # Adaptive baseline
            
        # Dispatch to specialized algorithm with perfect mapping
        detector = self._algo_map.get(algorithm, self._detect_sobel)
        return detector(gray_array, threshold)
    
    def _parse_algorithm(self, algorithm_name: str) -> EdgeDetector:
        """🧠 Convert algorithm identifier to enum with fuzzy matching.
        
        Maps text descriptors to algorithm types with intelligent
        pattern matching and perfect fallback strategy.
        
        Args:
            algorithm_name: Algorithm identifier string
            
        Returns:
            EdgeDetector: Mapped algorithm enum
        """
        name = algorithm_name.upper()
        try:
            # Direct mapping for exact match
            return EdgeDetector[name]
        except (KeyError, AttributeError):
            # Fuzzy matching for partial identifiers
            for algo in EdgeDetector:
                if algo.name.startswith(name[:3]):
                    return algo
            # Default to balanced algorithm with universal applicability
            return EdgeDetector.SOBEL
    
    def _convolve(self, channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """⚡ Apply convolution with zero-copy sliding window approach.
        
        Executes tensor convolution with maximum performance using
        stride-based window views and optimized tensor contraction.
        
        Args:
            channel: Single channel tensor
            kernel: Convolution kernel
            
        Returns:
            np.ndarray: Convolved tensor
        """
        # Edge handling with boundary replication
        padded = np.pad(channel, ((1, 1), (1, 1)), mode='edge')
        
        # Create sliding window view with zero memory overhead
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)
        
        # Execute tensor contraction for maximum performance
        return np.tensordot(windows, kernel, axes=([2, 3], [0, 1]))
    
    def _detect_sobel(self, gray_array: np.ndarray, threshold: int) -> GradientResult:
        """↔️↕️ Apply Sobel edge detection with balanced directionality.
        
        Maps visual discontinuities with the perfect balance of
        noise resistance and directional accuracy.
        
        Args:
            gray_array: Luminance tensor
            threshold: Edge sensitivity threshold
            
        Returns:
            GradientResult: Complete edge mapping with directional components
        """
        # Extract gradient components with zero memory overhead
        grad_x = self._convolve(gray_array, self._kernels["sobel_x"])
        grad_y = self._convolve(gray_array, self._kernels["sobel_y"])
        
        # Process with unified gradient pipeline
        return self._process_gradients(grad_x, grad_y, threshold)
    
    def _detect_prewitt(self, gray_array: np.ndarray, threshold: int) -> GradientResult:
        """🔊 Apply Prewitt detection for noise-resilient boundary mapping.
        
        Extracts edges with superior noise immunity and clean
        boundary detection in high-frequency visual domains.
        
        Args:
            gray_array: Luminance tensor
            threshold: Edge sensitivity threshold
            
        Returns:
            GradientResult: Complete edge mapping with directional components
        """
        # Extract gradient components
        grad_x = self._convolve(gray_array, self._kernels["prewitt_x"])
        grad_y = self._convolve(gray_array, self._kernels["prewitt_y"])
        
        return self._process_gradients(grad_x, grad_y, threshold)
    
    def _detect_scharr(self, gray_array: np.ndarray, threshold: int) -> GradientResult:
        """↗️↘️ Apply Scharr detection with perfect rotational symmetry.
        
        Maps discontinuities with superior angular consistency and
        precise gradient orientation for diagonal structures.
        
        Args:
            gray_array: Luminance tensor
            threshold: Edge sensitivity threshold
            
        Returns:
            GradientResult: Complete edge mapping with directional components
        """
        # Extract enhanced gradient components
        grad_x = self._convolve(gray_array, self._kernels["scharr_x"])
        grad_y = self._convolve(gray_array, self._kernels["scharr_y"])
        
        return self._process_gradients(grad_x, grad_y, threshold)
    
    def _detect_laplacian(self, gray_array: np.ndarray, threshold: int) -> GradientResult:
        """🔄 Apply Laplacian detection with second-order precision.
        
        Maps edge structures through second-order differential analysis
        with omnidirectional sensitivity and zero-crossing detection.
        
        Args:
            gray_array: Luminance tensor
            threshold: Edge sensitivity threshold
            
        Returns:
            GradientResult: Complete edge mapping with directional components
        """
        # Apply noise reduction for Laplacian stability
        if self.system_tier >= 1:
            gray_array = self._convolve(gray_array, self._kernels["gaussian"])
            
        # Apply Laplacian operator for omnidirectional detection
        laplacian = self._convolve(gray_array, self._kernels["laplacian"])
        
        # Get directional information from Sobel (provides gradient vectors)
        grad_x = self._convolve(gray_array, self._kernels["sobel_x"])
        grad_y = self._convolve(gray_array, self._kernels["sobel_y"])
        
        # Create result with Laplacian magnitude but Sobel direction
        magnitude = self._normalize_gradient(np.abs(laplacian), threshold)
        direction = np.arctan2(grad_y, grad_x)
        
        return {
            "magnitude": magnitude,
            "gradient_x": grad_x,
            "gradient_y": grad_y,
            "direction": direction
        }
    
    def _detect_canny(self, gray_array: np.ndarray, threshold: int) -> GradientResult:
        """✂️ Apply Canny detection with hysteresis and edge thinning.
        
        Implements complete Canny algorithm with non-maximum suppression,
        dual-threshold hysteresis, and perfect edge connectivity.
        
        Args:
            gray_array: Luminance tensor
            threshold: High threshold for hysteresis
            
        Returns:
            GradientResult: Complete edge mapping with directional components
        """
        # Start with Sobel gradients for directional information
        result = self._detect_sobel(gray_array, 0)
        grad = result["magnitude"].astype(np.float32)
        grad_x, grad_y = result["gradient_x"], result["gradient_y"]
        theta = result["direction"]
        
        # Quantize angles to 4 directions (0°, 45°, 90°, 135°)
        angle = (np.round(theta * (4/np.pi)) % 4).astype(np.uint8)
        
        # Non-maximum suppression for single-pixel edge precision
        height, width = grad.shape
        suppressed = np.zeros_like(grad)
        
        # Find significant gradient locations for sparse processing
        y_indices, x_indices = np.where(grad > 10)
        
        # Process edge pixels with direction-aware suppression
        for i, j in zip(y_indices, x_indices):
            # Skip boundary pixels
            if i == 0 or i == height-1 or j == 0 or j == width-1:
                continue
                
            # Select neighbors based on gradient direction
            if angle[i, j] == 0:      # Horizontal edges (─)
                n1, n2 = grad[i, j-1], grad[i, j+1]
            elif angle[i, j] == 1:    # Diagonal edges (↗)
                n1, n2 = grad[i+1, j-1], grad[i-1, j+1]
            elif angle[i, j] == 2:    # Vertical edges (│)
                n1, n2 = grad[i-1, j], grad[i+1, j]
            else:                     # Diagonal edges (↖)
                n1, n2 = grad[i-1, j-1], grad[i+1, j+1]
                
            # Retain only local maxima along gradient direction
            if grad[i, j] >= max(n1, n2):
                suppressed[i, j] = grad[i, j]
        
        # Hysteresis thresholding for edge continuity
        low_threshold = threshold // 2
        strong_edges = suppressed >= threshold
        weak_edges = (suppressed >= low_threshold) & (suppressed < threshold)
        
        # Initialize with strong edges
        edges = np.zeros_like(suppressed, dtype=np.uint8)
        edges[strong_edges] = 255
        
        # Connect weak edges to strong ones (8-connectivity)
        if np.any(weak_edges):
            # Find weak edge pixels
            weak_y, weak_x = np.where(weak_edges)
            
            # Check each weak edge for connection to strong edge
            for i, j in zip(weak_y, weak_x):
                # Get 3×3 neighborhood with boundary protection
                neighborhood = strong_edges[
                    max(0, i-1):min(height, i+2),
                    max(0, j-1):min(width, j+2)
                ]
                
                # Connect if adjacent to any strong edge
                if np.any(neighborhood):
                    edges[i, j] = 255
        
        return {
            "magnitude": edges,
            "gradient_x": grad_x,
            "gradient_y": grad_y,
            "direction": theta
        }
    
    def _process_gradients(self, 
                          grad_x: np.ndarray, 
                          grad_y: np.ndarray, 
                          threshold: int) -> GradientResult:
        """🔄 Transform gradient components into unified result.
        
        Processes raw gradient tensors into normalized magnitude and
        direction components with threshold application and scaling.
        
        Args:
            grad_x: Horizontal gradient tensor
            grad_y: Vertical gradient tensor
            threshold: Edge sensitivity threshold
            
        Returns:
            GradientResult: Complete edge components with perfect typing
        """
        # Calculate gradient magnitude with vectorized hypot
        grad = np.hypot(grad_x, grad_y)
        
        # Calculate direction for edge orientation (in radians)
        direction = np.arctan2(grad_y, grad_x)
        
        # Normalize and threshold the magnitude
        magnitude = self._normalize_gradient(grad, threshold)
        
        return {
            "magnitude": magnitude,    # Edge strength map
            "gradient_x": grad_x,      # Horizontal derivatives
            "gradient_y": grad_y,      # Vertical derivatives
            "direction": direction     # Angular direction map
        }
    
    def _normalize_gradient(self, gradient: np.ndarray, threshold: int) -> np.ndarray:
        """🔄 Normalize gradient with zero-division protection.
        
        Transforms raw gradient values to the standard 8-bit range
        with threshold application and perfect normalization.
        
        Args:
            gradient: Raw gradient tensor
            threshold: Edge sensitivity threshold
            
        Returns:
            np.ndarray: Normalized uint8 gradient tensor
        """
        # Handle empty gradient case with perfect resilience
        if gradient.size == 0 or np.max(gradient) <= 0:
            return np.zeros_like(gradient, dtype=np.uint8)
            
        # Normalize to [0, 255] range with perfect scaling
        normalized = (gradient * 255 / np.max(gradient)).clip(0, 255)
        
        # Apply threshold with zeroing of sub-threshold values
        thresholded = np.where(normalized < threshold, 0, normalized)
        
        # Re-normalize if significant values remain
        if np.max(thresholded) > 0:
            thresholded = (thresholded * 255 / np.max(thresholded)).clip(0, 255)
            
        return thresholded.astype(np.uint8)
    
    def clear_cache(self) -> None:
        """🧹 Release cached resources with thread safety."""
        with self._lock:
            self._cache.clear()
            
            # Clear function-level caches if decorated
            if hasattr(self.supersample_image, 'cache_clear'):
                self.supersample_image.cache_clear()


# Initialize the global processor instance
IMAGE_PROCESSOR = ImageProcessor.get_instance()

# Export core API functions with semantic naming
def supersample_image(image: Image.Image, scale_factor: int) -> Image.Image:
    """🔍 Upscale image matrix with adaptive quality enhancement."""
    return IMAGE_PROCESSOR.supersample_image(image, scale_factor)

def rgb_to_gray(image_array: np.ndarray) -> np.ndarray:
    """🌓 Transform RGB to perceptually-weighted grayscale."""
    return IMAGE_PROCESSOR.rgb_to_gray(image_array)

def detect_edges(gray_array: np.ndarray, algorithm: str = "sobel") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """🔪 Extract edge tensors with algorithm-specific optimization.
    
    Args:
        gray_array: Grayscale image tensor
        algorithm: Detection algorithm name
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Magnitude, grad_x, grad_y
    """
    result = IMAGE_PROCESSOR.detect_edges(gray_array, algorithm)
    return result["magnitude"], result["gradient_x"], result["gradient_y"]


# ╔══════════════════════════════════════════════════════════════╗
# ║ 🎞️ Streaming Intelligence                                   ║
# ╚══════════════════════════════════════════════════════════════╝

class StreamEngine:
    """Multi-dimensional adaptive stream processing system.
    
    Processes video streams with contextual awareness, dynamic quality adaptation,
    and network resilience. Handles YouTube URLs, local files, and camera inputs
    with automatic parameter optimization based on system capabilities.
    
    Attributes:
        _stream_cache: Cache mapping of URLs to streaming addresses and timestamps
        _cache_ttl: Cache entry lifetime in seconds
        _cache_max_size: Maximum number of cache entries to maintain
    """
    _stream_cache: Dict[str, Tuple[str, float]] = {}
    _cache_ttl: int = 3600  # 1 hour validity
    _cache_max_size: int = 100
    
    @classmethod
    def extract_stream_url(cls, youtube_url: str, resolution: Optional[int] = None) -> str:
        """⚡ Extract adaptive-quality stream URL with dimensional resilience matrix.
        
        Applies progressive fallback strategies with temporal caching and
        quality optimization based on system capabilities. All operations
        maintain perfect error isolation with zero cascading failures.
        
        Args:
            youtube_url: YouTube video URL or ID
            resolution: Preferred vertical resolution (None=auto-select)
            
        Returns:
            str: Direct streaming URL with optimal format
            
        Raises:
            DependencyError: If yt-dlp is not available
            StreamExtractionError: When extraction fails after recovery attempts
        """
        if not HAS_YT_DLP:
            raise DependencyError("yt-dlp", "pip install yt-dlp", "YouTube streaming")
        
        # Efficient cache lookup with integrated TTL validation
        cache_key = f"{youtube_url}:{resolution}"
        current_time = time.time()
        cls._prune_cache(current_time)
        
        if cache_key in cls._stream_cache:
            url, timestamp = cls._stream_cache[cache_key]
            if current_time - timestamp < cls._cache_ttl:
                return url
        
        # Dynamic resolution selection based on system capabilities
        actual_resolution = resolution or cls._determine_optimal_resolution()
        format_spec = f'best[height<={actual_resolution}][ext=mp4]'
        
        ydl_opts = {
            'format': format_spec,
            'quiet': True,
            'skip_download': True,
            'no_warnings': True,
            'socket_timeout': 10,
        }
        
        # Progressive fallback with adaptive retry strategy
        for retry in range(3):
            try:
                # 🛡️ Dimension-safe status display with perfect fallbacks
                status_msg = f"🔍 Extracting stream{'.' * (retry + 1)}"
                if HAS_RICH and CONSOLE:
                    try:
                        print(f"\r{status_msg}...", end="", flush=True)  # Stable fallback first
                    except Exception:
                        pass  # Silent recovery
                else:
                    print(f"\r{status_msg}...", end="", flush=True)
                    
                # Perform extraction with error isolation barrier
                extraction_result = cls._perform_extraction(youtube_url, ydl_opts)
                
                # Clear status line with universal compatibility
                print("\r" + " " * 50 + "\r", end="", flush=True)
                    
                if extraction_result.url:
                    # Cache successful result with temporal binding
                    cls._stream_cache[cache_key] = (extraction_result.url, current_time)
                    
                    # Success indicator with minimal footprint
                    if not HAS_RICH:
                        print(f"✓ Stream extracted: {extraction_result.format}", flush=True)
                    
                    return extraction_result.url
                    
                # Format degradation for retry with perfect fallbacks
                ydl_opts['format'] = 'best[height<=360]' if retry == 0 else 'worst'
                
            except Exception as e:
                # Clear status line with universal compatibility
                print("\r" + " " * 50 + "\r", end="", flush=True)
                
                error_category = cls._categorize_extraction_error(e)
                
                if retry < 2:
                    # Exponential backoff with format adaptation
                    if not HAS_RICH:
                        print(f"⚠️ Extraction attempt {retry+1} failed: {error_category}. Retrying...", flush=True)
                        
                    time.sleep(1 * (2**retry))
                    if error_category in ('network', 'timeout'):
                        ydl_opts['socket_timeout'] = 20
                        ydl_opts['format'] = 'best[height<=360]' if retry == 0 else 'worst'
                else:
                    if not HAS_RICH:
                        print(f"❌ Extraction failed: {error_category}", flush=True)
                        
                    raise StreamExtractionError(
                        f"Failed to extract stream: {error_category}", 
                        original=e, 
                        category=error_category
                    )
                
        raise StreamExtractionError("Stream extraction failed after multiple attempts")

    @classmethod
    def _prune_cache(cls, current_time: float) -> None:
        """🧹 Remove expired and excess cache entries with minimal iterations.
        
        Ensures perfect memory efficiency with time complexity O(n log n)
        in worst case, O(1) in common case where cache is within limits.
        
        Args:
            current_time: Current timestamp for TTL comparison
        """
        # First remove expired entries with zero overhead
        expired_keys = [k for k, (_, ts) in cls._stream_cache.items() 
                      if current_time - ts > cls._cache_ttl]
        
        for key in expired_keys:
            cls._stream_cache.pop(key, None)
            
        # Then enforce size limit if still needed
        if len(cls._stream_cache) > cls._cache_max_size:
            # Keep only the newest entries
            sorted_items = sorted(
                cls._stream_cache.items(), 
                key=lambda item: item[1][1],  # Sort by timestamp
                reverse=True  # Newest first
            )
            # Reset cache with only newest entries
            cls._stream_cache = dict(sorted_items[:cls._cache_max_size])

    @staticmethod
    def _determine_optimal_resolution() -> int:
        """🔍 Select optimal resolution based on system capabilities and terminal size.
        
        Applies multidimensional heuristics including performance tier,
        terminal dimensions, and network conditions for perfect balance
        between quality and performance.
        
        Returns:
            int: Vertical resolution in pixels
        """
        system_tier = ENV.capabilities.get("performance_tier", 1)
        terminal_height = ENV.terminal.get("height", 24)
        
        # Tiered resolution selection with dimensional awareness
        if system_tier >= 3:
            return 1080 if terminal_height > 60 else 720
        elif system_tier >= 2 and terminal_height > 40:
            return 720
        elif system_tier >= 1 and terminal_height > 30:
            return 480
        else:
            return 360

    @staticmethod
    def _categorize_extraction_error(error: Exception) -> str:
        """📊 Categorize extraction errors for strategic retry decisions.
        
        Maps error patterns to semantic categories with zero ambiguity
        for precise recovery strategy selection.
        
        Args:
            error: Exception from extraction attempt
            
        Returns:
            str: Error category identifier
        """
        error_str = str(error).lower()
        
        # Pattern-based error classification with perfect specificity
        if "429" in error_str:
            return "rate_limited"
        elif "403" in error_str:
            return "access_denied"
        elif any(x in error_str for x in ("blocked", "not available", "country")):
            return "geo_restricted"
        elif any(x in error_str for x in ("timeout", "timed out", "connection")):
            return "network"
        return "general"

    @staticmethod
    def _perform_extraction(url: str, options: Dict[str, Any]) -> VideoInfo:
        """🧬 Extract video information with structured result handling.
        
        Performs clean extraction with perfect error isolation and
        comprehensive metadata normalization.
        
        Args:
            url: Video URL to process
            options: YoutubeDL options dictionary
            
        Returns:
            VideoInfo: Structured video metadata
        """
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
            
            return VideoInfo(
                url=info.get("url"),
                title=info.get("title", "Unknown video"),
                duration=info.get("duration"),
                format=info.get("format_id", "unknown"),
                width=info.get("width"),
                height=info.get("height"),
                fps=info.get("fps")
            )

    @classmethod
    def process_video_stream(
        cls,
        source: Union[str, int, Path],
        scale_factor: int = 2,
        block_width: int = 8,
        block_height: int = 8,
        edge_threshold: int = 50,
        gradient_str: Optional[str] = None,
        color: bool = True,
        fps: int = 15,
        enhanced_edges: bool = True,
        show_stats: bool = True,
        adaptive_quality: bool = True,
        border: bool = True
    ) -> None:
        """⚡ Process video stream with adaptive rendering and performance tuning.
        
        Streams content from various sources (files, URLs, cameras) with
        dynamic quality adjustment and performance monitoring. Applies
        dimensional compression with recursive quality optimization
        for perfect balance of visual fidelity and performance.
        
        Args:
            source: File path, YouTube URL, or camera index
            scale_factor: Detail enhancement factor (1-4)
            block_width: Character cell width in pixels
            block_height: Character cell height in pixels
            edge_threshold: Edge detection sensitivity (0-100)
            gradient_str: Custom character gradient (None=auto)
            color: Enable terminal colors
            fps: Target frames per second
            enhanced_edges: Use directional edge detection
            show_stats: Display performance metrics
            adaptive_quality: Dynamically adjust quality for performance
            border: Add decorative frame
            
        Raises:
            DependencyError: If OpenCV is not available
            IOError: If source cannot be opened
        """
        if not HAS_CV2:
            raise DependencyError("opencv-python", "pip install opencv-python", "Video processing")
            
        # Initialize stream with integrated error handling
        try:
            # Normalize source and open capture
            stream_url = cls._resolve_source(source)
            capture = cv2.VideoCapture(stream_url)
            
            if not capture.isOpened():
                raise IOError(f"Failed to open source: {source}")
                
            # Extract video metadata
            video_info = VideoInfo.from_capture(capture, str(source), 
                                               "youtube" if isinstance(source, str) and 
                                               ("youtu" in source or "://y" in source) else "file")
        except Exception as e:
            cls._handle_stream_error(e, 0)
            return
        
        # Setup rendering parameters with environment optimization
        params = RenderParameters(
            scale=scale_factor,
            width=block_width,
            height=block_height,
            threshold=edge_threshold,
            optimal_width=block_width,
            optimal_height=block_height
        )
        
        # Performance monitoring
        metrics = StreamMetrics()
        frame_buffer = FrameBuffer(capacity=2)
        
        # Terminal UI initialization
        renderer = FrameRenderer(
            terminal_width=ENV.terminal["width"],
            terminal_height=ENV.terminal["height"],
            gradient=gradient_str or get_enhanced_gradient_chars(),
            border=border and ENV.capabilities["unicode"],
            unicode_supported=ENV.capabilities["unicode"]
        )
        
        # Display stream information with perfect resilience
        title = f"🎬 Streaming: {video_info.title}"
        info = [f"Source: {source}"]
        if video_info.width and video_info.height:
            info.append(f"Resolution: {video_info.width}×{video_info.height}")
        if video_info.fps:
            info.append(f"Original FPS: {video_info.fps:.1f}")
            
        # 🛡️ Safe output with perfect fallbacks
        try:
            if HAS_RICH and CONSOLE:
                try:
                    CONSOLE.print(Panel("\n".join(info), title=title, border_style="blue"))
                except Exception:
                    # Fallback to plain text on Rich failure
                    print(f"\n{title}")
                    print("\n".join(info))
                    print("=" * 40)
            else:
                print(f"\n{title}")
                print("\n".join(info))
                print("=" * 40)
        except Exception:
            # Ultimate fallback - guaranteed to work
            print(f"\nStreaming: {video_info.title}")
            
        # Performance thresholds for adaptive quality
        render_thresholds = RenderThresholds.from_target_fps(fps)
        frame_time = 1.0 / fps  # Target frame time in seconds
        
        # Clear screen before streaming
        renderer.clear_screen()
        
        # Process frames with error handling
        try:
            running = True
            last_render_time = time.time()
            frames_processed = 0
            
            while running:
                start_time = time.time()
                
                # Frame capture
                ret, frame = capture.read()
                if not ret:
                    break  # End of stream
                    
                # Auto-rotate based on EXIF orientation
                if hasattr(frame, 'shape') and len(frame.shape) >= 2:
                    if frame.shape[0] > frame.shape[1] and frame.shape[0] > 720:
                        # Portrait video - transpose for better terminal fit
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                
                # Convert to PIL for processing
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Process frame with timing
                render_start = time.time()
                art = image_to_unicode_art(
                    pil_image,
                    scale_factor=params.scale,
                    block_width=params.width,
                    block_height=params.height,
                    edge_threshold=params.threshold,
                    gradient_str=gradient_str,
                    color=color,
                    enhanced_edges=enhanced_edges
                )
                render_time = time.time() - render_start
                
                # Update metrics
                metrics.record_render(render_time)
                metrics.record_frame()
                metrics.update_fps()
                
                # Render complete frame with border and stats
                frame_lines = renderer.render_frame(
                    art=art,
                    source_name=video_info.title or str(source),
                    metrics=metrics,
                    params=params,
                    show_stats=show_stats
                )
                
                # Add to buffer
                frame_buffer.add(frame_lines)
                
                # Throttle display based on target frame rate
                elapsed = time.time() - last_render_time
                if elapsed >= frame_time or frames_processed == 0:
                    # Display latest frame
                    renderer.clear_screen()
                    renderer.display_frame(frame_buffer.get_latest())
                    last_render_time = time.time()
                    frames_processed += 1
                    
                    # Adapt quality if enabled
                    if adaptive_quality and frames_processed % 5 == 0:
                        params.adjust_quality(render_time * 1000, render_thresholds)
                
                # Control frame rate with precise timing
                elapsed_total = time.time() - start_time
                sleep_time = max(0, frame_time - elapsed_total)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    metrics.record_dropped()
                    
        except KeyboardInterrupt:
            # Clean exit with statistics
            renderer.clear_screen()
            
            # 🛡️ Safe stats display with perfect fallbacks
            try:
                if HAS_RICH and CONSOLE:
                    try:
                        CONSOLE.print("\n[bold green]✓ Stream completed![/bold green]")
                        
                        perf_stats = metrics.get_stats()
                        
                        # Create stats table with resilient approach
                        try:
                            stats_table = Table(title="📊 Performance Summary")
                            stats_table.add_column("Metric", style="cyan")
                            stats_table.add_column("Value", style="green")
                            
                            stats_table.add_row("Frames processed", str(perf_stats["total_frames"]))
                            stats_table.add_row("Average FPS", f"{perf_stats['avg_fps']:.2f}")
                            stats_table.add_row("Render time", f"{perf_stats['avg_render_time']:.1f}ms")
                            stats_table.add_row("Final quality", f"{params.quality_level.name}")
                            if perf_stats["dropped_frames"] > 0:
                                stats_table.add_row("Dropped frames", 
                                                f"{perf_stats['dropped_frames']} ({perf_stats['drop_ratio']*100:.1f}%)")
                            
                            CONSOLE.print(stats_table)
                        except Exception:
                            # Fallback to direct print on Rich table error
                            raise RuntimeError("Stats table creation failed")
                    except Exception:
                        # Fallback to plain text on any Rich error
                        raise RuntimeError("Rich console failed")
                else:
                    raise RuntimeError("Rich not available")
            except Exception:
                # Ultimate fallback - guaranteed to work
                print("\n✓ Stream completed!")
                print(f"Frames processed: {metrics.frames_processed}")
                print(f"Average FPS: {metrics.current_fps:.2f}")
                print(f"Render time: {metrics.average_render_time:.1f}ms")
                
        except Exception as e:
            cls._handle_stream_error(e, metrics.frames_processed)
            
        finally:
            if capture is not None:
                capture.release()

    @staticmethod
    def _resolve_source(source: Union[str, int, Path]) -> Union[str, int]:
        """🔄 Resolve stream source with YouTube URL extraction when needed.
        
        Performs multi-stage source normalization with perfect fallbacks
        and comprehensive format detection.
        
        Args:
            source: Original source identifier
            
        Returns:
            Stream URL or device identifier
        """
        # Handle Path objects
        if isinstance(source, Path):
            return str(source)
            
        # Handle numeric camera indices
        if isinstance(source, int):
            return source
            
        # Handle string URLs/paths
        source_str = str(source)
        
        # YouTube URL detection and resolution
        if any(x in source_str.lower() for x in ("youtube.com", "youtu.be", "yt.be")):
            try:
                return StreamEngine.extract_stream_url(source_str)
            except Exception as e:
                # Safe error reporting with guaranteed output
                print(f"\n⚠️ YouTube extraction failed: {type(e).__name__}")
                print(f"💡 Attempting direct access instead...")
                    
        return source_str
        
    @staticmethod
    def _handle_stream_error(error: Exception, frames_processed: int) -> None:
        """🛡️ Display user-friendly error with context and recovery guidance.
        
        Provides semantically rich error information with perfect format
        resilience and actionable recovery guidance.
        
        Args:
            error: Exception that occurred
            frames_processed: Number of frames successfully processed
        """
        if frames_processed > 0:
            # Non-fatal error after some successful processing
            message = "Stream interrupted"
            style = "yellow"
        else:
            # Fatal error during initialization
            message = "Failed to process stream"
            style = "red bold"
        
        # 🛡️ Safe error display with perfect fallbacks
        try:
            if HAS_RICH and CONSOLE:
                try:
                    CONSOLE.print(f"[{style}]🚫 {message}: {type(error).__name__}[/{style}]")
                    
                    if frames_processed == 0:
                        # More detailed error for startup failures
                        CONSOLE.print(f"[dim]{str(error)}[/dim]")
                        
                        if isinstance(error, cv2.error):
                            CONSOLE.print("[yellow]💡 This may be due to an unsupported video format or codec.[/yellow]")
                        elif "connection" in str(error).lower():
                            CONSOLE.print("[yellow]💡 Check your network connection or URL.[/yellow]")
                except Exception:
                    # Fall through to plain text on any Rich error
                    raise RuntimeError("Rich console failed")
            else:
                raise RuntimeError("Rich not available")
        except Exception:
            # Ultimate fallback - guaranteed to work
            print(f"\n🚫 {message}: {type(error).__name__}")
            print(f"  {str(error)}")
            
            if frames_processed == 0:
                if isinstance(error, cv2.error):
                    print("💡 This may be due to an unsupported video format or codec.")
                elif "connection" in str(error).lower():
                    print("💡 Check your network connection or URL.")

# Mapping to Legacy Functions For Backward Compatibility
def process_video_stream(
    source: Union[str, int, Path],
    scale_factor: int = 2,
    block_width: int = 8,
    block_height: int = 8,
    edge_threshold: int = 50,
    gradient_str: Optional[str] = None,
    color: bool = True,
    fps: int = 15,
    enhanced_edges: bool = True,
    show_stats: bool = True,
    adaptive_quality: bool = True,
    border: bool = True
) -> None:
    """⚡ Process video streams with multidimensional rendering and adaptive quality.

    Provides high-performance stream processing with intelligent buffering, 
    concurrent rendering, and real-time quality adaptation based on system capabilities.
    Perfect integration of advanced algorithms with zero-overhead processing.
    
    Args:
        source: Video file path, YouTube URL, camera index, or Path object
        scale_factor: Detail enhancement multiplier (1-4)
        block_width: Character cell width in pixels
        block_height: Character cell height in pixels
        edge_threshold: Edge detection sensitivity (0-255)
        gradient_str: Custom density gradient characters (None=auto-select)
        color: Whether to enable ANSI color output
        fps: Target frames per second
        enhanced_edges: Whether to use directional edge characters
        show_stats: Whether to display performance metrics
        adaptive_quality: Whether to auto-adjust quality for performance
        border: Whether to add decorative frame around content
    
    Raises:
        DependencyError: If required dependencies are missing
        IOError: If the source cannot be opened
    """
    StreamEngine.process_video_stream(
        source=source,
        scale_factor=scale_factor,
        block_width=block_width,
        block_height=block_height,
        edge_threshold=edge_threshold,
        gradient_str=gradient_str,
        color=color,
        fps=fps,
        enhanced_edges=enhanced_edges,
        show_stats=show_stats,
        adaptive_quality=adaptive_quality,
        border=border
    )

@dataclass(frozen=True)
class RenderThresholds:
    """Adaptive quality thresholds based on render time performance.
    
    Attributes:
        reduce_ms: Threshold in ms above which quality should be reduced
        improve_ms: Threshold in ms below which quality can be improved
    """
    reduce_ms: float  # Upper threshold to trigger quality reduction
    improve_ms: float  # Lower threshold to trigger quality improvement
    
    @classmethod
    def from_target_fps(cls, target_fps: float, reduce_ratio: float = 0.9, 
                       improve_ratio: float = 0.6) -> 'RenderThresholds':
        """Create optimal thresholds from target FPS with balanced margins.
        
        Args:
            target_fps: Desired frames per second
            reduce_ratio: Percentage of frame budget that triggers quality reduction
            improve_ratio: Percentage of frame budget that allows quality improvement
            
        Returns:
            RenderThresholds: Calculated thresholds for adaptive rendering
        """
        frame_budget_ms = (1000.0 / target_fps)
        return cls(
            reduce_ms=frame_budget_ms * reduce_ratio,
            improve_ms=frame_budget_ms * improve_ratio
        )


class StreamMetrics:
    """High-precision performance metrics with statistical analysis.
    
    Thread-safe performance tracking for realtime visualization with
    rolling statistics and comprehensive analytics.
    
    Attributes:
        frames_processed: Total successfully rendered frames
        current_fps: Most recently calculated frames per second
        average_render_time: Mean render time in milliseconds
    """
    
    def __init__(self, sample_size: int = 30) -> None:
        """Initialize performance tracking with optimized sample size.
        
        Args:
            sample_size: Maximum samples for rolling statistics
        """
        self._lock = threading.RLock()
        self.frames_processed: int = 0
        self.dropped_frames: int = 0
        self.start_time: float = time.time()
        self.current_fps: float = 0.0
        
        # Performance tracking with bounded collections
        self._render_times = collections.deque[float](maxlen=sample_size)
        self._fps_samples = collections.deque[float](maxlen=sample_size)
        self._last_fps_time = time.time()
        self._frames_since_fps_update = 0
        
    def update_fps(self) -> float:
        """Calculate current FPS with time-weighted accuracy.
        
        Returns:
            float: Current frames per second
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_fps_time
            
            # Only update when significant time has passed
            if elapsed >= 1.0 and self._frames_since_fps_update > 0:
                self.current_fps = self._frames_since_fps_update / elapsed
                self._fps_samples.append(self.current_fps)
                self._last_fps_time = now
                self._frames_since_fps_update = 0
                
            return self.current_fps
        
    def record_render(self, duration: float) -> None:
        """Record frame rendering duration with millisecond precision.
        
        Args:
            duration: Render time in seconds
        """
        with self._lock:
            self._render_times.append(duration * 1000)
        
    def record_frame(self) -> None:
        """Record successful frame processing with thread safety."""
        with self._lock:
            self.frames_processed += 1
            self._frames_since_fps_update += 1
        
    def record_dropped(self) -> None:
        """Record frame drop for performance analytics."""
        with self._lock:
            self.dropped_frames += 1
    
    @property
    def average_render_time(self) -> float:
        """Calculate mean render time with optimized algorithm.
        
        Returns:
            float: Average render time in milliseconds
        """
        with self._lock:
            return (sum(self._render_times) / len(self._render_times) 
                   if self._render_times else 0.0)
    
    @property
    def effective_fps(self) -> float:
        """Calculate overall FPS across entire runtime.
        
        Returns:
            float: Overall frames per second
        """
        with self._lock:
            duration = time.time() - self.start_time
            return self.frames_processed / max(0.001, duration)
    
    @property
    def drop_ratio(self) -> float:
        """Calculate percentage of frames dropped.
        
        Returns:
            float: Ratio of dropped frames (0.0-1.0)
        """
        with self._lock:
            total = self.frames_processed + self.dropped_frames
            return self.dropped_frames / max(1, total)
    
    def get_stats(self) -> PerformanceStats:
        """Generate comprehensive performance analytics.
        
        Returns:
            PerformanceStats: Complete metrics dictionary
        """
        with self._lock:
            # Use fast mean calculation with bounds checking
            avg_fps = (sum(self._fps_samples) / len(self._fps_samples) 
                      if self._fps_samples else 0.0)
            
            # Calculate stability as inverse of coefficient of variation
            stability = 1.0
            if len(self._render_times) > 1:
                try:
                    mean = self.average_render_time
                    if mean > 0:
                        # Calculate variance without full sample replication
                        variance = sum((t - mean)**2 for t in self._render_times) / len(self._render_times)
                        # Normalize to 0-1 scale with saturation
                        cv = min(1.0, math.sqrt(variance) / mean)
                        stability = 1.0 - cv
                except (ZeroDivisionError, OverflowError):
                    pass
            
            return {
                "avg_render_time": self.average_render_time,
                "avg_fps": avg_fps,
                "effective_fps": self.effective_fps,
                "total_frames": self.frames_processed,
                "dropped_frames": self.dropped_frames,
                "drop_ratio": self.drop_ratio,
                "stability": stability
            }


class RenderParameters:
    """Adaptive rendering parameters with dimensional quality management.
    
    Provides intelligent quality scaling across multiple dimensions
    with optimized parameter sets for different performance targets.
    
    Attributes:
        scale: Detail enhancement factor (1-4)
        width: Block width in pixels
        height: Block height in pixels
        threshold: Edge detection sensitivity (0-100)
        quality_level: Current quality preset (MINIMAL-MAXIMUM)
    """
    
    def __init__(
        self, 
        scale: int,
        width: int,
        height: int,
        threshold: int,
        optimal_width: int,
        optimal_height: int,
        quality_level: QualityLevel = QualityLevel.STANDARD
    ) -> None:
        """Initialize with balanced default parameters and constraints.
        
        Args:
            scale: Detail enhancement factor
            width: Character cell width
            height: Character cell height
            threshold: Edge sensitivity
            optimal_width: Reference width for quality scaling
            optimal_height: Reference height for quality scaling
            quality_level: Initial quality preset
        """
        self._lock = threading.RLock()
        
        # Normalize parameters with bounds protection
        self.scale = max(1, min(4, scale))
        self.width = max(4, width)
        self.height = max(2, height)
        self.threshold = max(30, min(80, threshold))
        
        # Quality reference points
        self.optimal_width = max(4, optimal_width)
        self.optimal_height = max(2, optimal_height)
        self.quality_level = quality_level
        self.frames_since_adjustment: int = 0
        
        # Parameter mappings for efficient quality transitions
        self._quality_map = {
            QualityLevel.MINIMAL: (1, 4, 2),  # Fastest rendering
            QualityLevel.LOW: (max(1, self.scale - 1), 0, 0),
            QualityLevel.STANDARD: (self.scale, 0, 0),  # Reference quality
            QualityLevel.HIGH: (min(4, self.scale + 1), -2, -1),
            QualityLevel.MAXIMUM: (min(4, self.scale + 1), -4, -2)  # Highest detail
        }
        
    def adjust_quality(self, render_time: float, 
                      thresholds: Union[RenderThresholds, Dict[str, float]]) -> bool:
        """Adjust quality based on performance with hysteresis.
        
        Intelligently balances quality and performance by adjusting
        multiple parameters simultaneously based on render times.
        
        Args:
            render_time: Current render time in milliseconds
            thresholds: Performance boundaries for adjustments
            
        Returns:
            bool: Whether quality was changed
        """
        with self._lock:
            # Cache previous level for change detection
            previous_level = self.quality_level
            
            # Extract thresholds with protocol compatibility
            reduce_ms = (thresholds.reduce_ms if isinstance(thresholds, RenderThresholds) 
                        else thresholds['reduce'])
            improve_ms = (thresholds.improve_ms if isinstance(thresholds, RenderThresholds) 
                        else thresholds['improve'])
            
            # Apply quality transitions with bounds protection
            if render_time > reduce_ms and self.quality_level > QualityLevel.MINIMAL:
                self.quality_level = QualityLevel(self.quality_level - 1)
            elif render_time < improve_ms and self.quality_level < QualityLevel.MAXIMUM:
                self.quality_level = QualityLevel(self.quality_level + 1)
            
            # Apply parameter changes if quality changed
            changed = previous_level != self.quality_level
            if changed:
                self._apply_quality_parameters()
            
            # Track frames since last adjustment
            self.frames_since_adjustment = 0 if changed else self.frames_since_adjustment + 1
            return changed
        
    def _apply_quality_parameters(self) -> None:
        """Update rendering parameters for current quality level."""
        with self._lock:
            # Get parameters for selected quality level
            new_scale, width_delta, height_delta = self._quality_map[self.quality_level]
            
            # Apply parameters with proportional scaling
            self.scale = new_scale
            self.width = max(4, self.optimal_width + width_delta)
            self.height = max(2, self.optimal_height + height_delta)
            
            # Adjust threshold for detail preservation
            if self.quality_level >= QualityLevel.HIGH:
                self.threshold = max(30, self.threshold - 10)
            elif self.quality_level <= QualityLevel.LOW:
                self.threshold = min(80, self.threshold + 10)


class FrameRenderer:
    """🎭 Dimensional frame renderer with adaptive formatting and concurrent support.
    
    Transforms raw art matrices into perfectly structured visual frameworks with
    intelligent border styling, title display, and performance metrics visualization.
    Every component precision-engineered for maximum contextual integrity.
    
    Attributes:
        terminal_width (int): Available width for rendering
        terminal_height (int): Available height for rendering
        gradient (str): Character density gradient
        border (bool): Whether to display decorative borders
        unicode_supported (bool): Whether terminal supports Unicode
    """
    
    def __init__(
        self, 
        terminal_width: int, 
        terminal_height: int,
        gradient: str,
        border: bool = True,
        unicode_supported: bool = True
    ) -> None:
        """Initialize renderer with environment configuration and optimized caching.
        
        Args:
            terminal_width: Available terminal width in characters
            terminal_height: Available terminal height in lines
            gradient: Character gradient for density rendering
            border: Whether to display decorative border
            unicode_supported: Whether terminal supports unicode characters
        """
        self.terminal_width = max(40, terminal_width)
        self.terminal_height = max(10, terminal_height)
        self.gradient = gradient
        self.border = border
        self.unicode_supported = unicode_supported
        self.border_chars = self._get_border_chars(border, unicode_supported)
        
        # High-performance LRU caches for rendered components
        self._title_cache: Dict[str, str] = {}
        self._stats_cache: Dict[str, str] = {}
        self._lock = threading.RLock()
    
    def _get_border_chars(self, border: bool, unicode_supported: bool) -> Dict[str, str]:
        """Get optimal border character set based on terminal capabilities.
        
        Args:
            border: Whether borders are enabled
            unicode_supported: Whether Unicode is supported
            
        Returns:
            Dict[str, str]: Border character mapping
        """
        return {
            "top_left": "╔" if border and unicode_supported else "+",
            "top_right": "╗" if border and unicode_supported else "+",
            "bottom_left": "╚" if border and unicode_supported else "+",
            "bottom_right": "╝" if border and unicode_supported else "+",
            "horizontal": "═" if border and unicode_supported else "-",
            "vertical": "║" if border and unicode_supported else "|",
        }
            
    def render_frame(
        self, 
        art: List[str], 
        source_name: str,
        metrics: StreamMetrics,
        params: RenderParameters,
        show_stats: bool = True
    ) -> List[str]:
        """Render complete frame with content, border and statistics.
        
        Args:
            art: Unicode art content lines
            source_name: Source identifier for title
            metrics: Performance metrics for statistics
            params: Current rendering parameters
            show_stats: Whether to display performance statistics
            
        Returns:
            List[str]: Fully rendered frame lines
        """
        if not art:
            return []
            
        with self._lock:
            frame = []
            content_width = len(art[0]) if art else 0
            
            # Add title with cached formatting
            if self.border:
                cache_key = f"{source_name}:{content_width}"
                title_line = self._title_cache.get(cache_key)
                if not title_line:
                    title_line = self._format_title(source_name, content_width)
                    self._title_cache[cache_key] = title_line
                    # Bound cache size
                    if len(self._title_cache) > 50:
                        self._title_cache.pop(next(iter(self._title_cache)))
                frame.append(title_line)
            
            # Add content directly
            frame.extend(art)
            
            # Add statistics with caching
            if show_stats and metrics.frames_processed > 0:
                if self.border:
                    # Bottom border
                    border_line = f"{self.border_chars['bottom_left']}{self.border_chars['horizontal'] * content_width}{self.border_chars['top_right']}"
                    frame.append(border_line)
                    
                    # Stats with cached formatting
                    cache_key = f"{metrics.current_fps:.1f}:{metrics.average_render_time:.1f}:{params.quality_level}"
                    stats_line = self._stats_cache.get(cache_key)
                    if not stats_line:
                        stats_line = self._format_stats(metrics, params, content_width)
                        self._stats_cache[cache_key] = stats_line
                        # Bound cache size
                        if len(self._stats_cache) > 50:
                            self._stats_cache.pop(next(iter(self._stats_cache)))
                    
                    frame.append(f"{self.border_chars['vertical']} {stats_line} {self.border_chars['vertical']}")
                    frame.append(f"{self.border_chars['bottom_left']}{self.border_chars['horizontal'] * content_width}{self.border_chars['bottom_right']}")
                else:
                    frame.append(f"FPS: {metrics.current_fps:.1f} | Render: {metrics.average_render_time:.1f}ms | Q: {params.quality_level}/4")
            
            return frame
        
    def _format_title(self, source_name: str, width: int) -> str:
        """Format title bar with centered source name and borders.
        
        Args:
            source_name: Source identifier
            width: Content width
            
        Returns:
            str: Formatted title line
        """
        basename = os.path.basename(str(source_name))
        title = f" {basename} "
        padding = max(0, width - len(title))
        left_pad, right_pad = padding // 2, padding - (padding // 2)
        
        return f"{self.border_chars['top_left']}{self.border_chars['horizontal'] * left_pad}{title}{self.border_chars['horizontal'] * right_pad}{self.border_chars['top_right']}"
    
    def _format_stats(self, metrics: StreamMetrics, params: RenderParameters, width: int) -> str:
        """Format performance statistics with quality indicators.
        
        Args:
            metrics: Performance metrics to display
            params: Current rendering parameters
            width: Available width for stats
            
        Returns:
            str: Formatted stats with quality indicators
        """
        # Compact stats formatting with visual quality indicator
        fps = f"FPS: {metrics.current_fps:.1f}"
        render = f"Render: {metrics.average_render_time:.1f}ms"
        quality = f"Quality: {params.quality_level}/4"
        indicator = '●' * params.quality_level + '○' * (4 - params.quality_level)
        
        stats = f"{fps} | {render} | {quality} | {indicator}"
        return stats[:width-3] + "..." if len(stats) > width else stats.ljust(width)
    
    def clear_screen(self) -> None:
        """🧹 Clear terminal with platform-specific approach."""
        # Use direct system calls for most reliable clearing
        try:
            if os.name == 'posix':  # Linux/Mac
                os.system('clear')
            elif os.name == 'nt':   # Windows
                os.system('cls')
            else:
                # Simple newlines as safe fallback
                print("\n" * 100, end="")
        except Exception:
            # Ultimate fallback
            print("\n" * 100, end="")
    
    def display_frame(self, frame_lines: List[str]) -> None:
        """Display frame with optimized I/O operations.
        
        Args:
            frame_lines: Rendered frame lines to display
        """
        if frame_lines:
            print("\n".join(frame_lines), flush=True)

class FrameBuffer:
    """Thread-safe frame buffer with automatic capacity management.
    
    Provides concurrent FIFO buffer operations with intelligent frame management
    to ensure optimal rendering performance under varying system loads.
    
    Attributes:
        capacity (int): Maximum frames to store
        frames (collections.deque): Buffered frame data
    """
    
    def __init__(self, capacity: int = 2) -> None:
        """Initialize buffer with optimized capacity.
        
        Args:
            capacity: Maximum number of frames to store
        """
        self._frames: collections.deque[List[str]] = collections.deque(maxlen=capacity)
        self._lock = threading.RLock()
        self._dropped_frames = 0
        
    def add(self, frame: List[str]) -> None:
        """Add frame to buffer with thread-safety.
        
        Args:
            frame: Frame lines to add
        """
        with self._lock:
            self._frames.append(frame)
            
    def get_latest(self) -> List[str]:
        """Get most recent frame with empty fallback.
        
        Returns:
            List[str]: Latest frame or empty list
        """
        with self._lock:
            return self._frames[-1] if self._frames else []
            
    def clear(self) -> None:
        """Clear all frames with thread-safety."""
        with self._lock:
            self._frames.clear()
            
    @property
    def size(self) -> int:
        """Get current buffer size.
        
        Returns:
            int: Number of frames in buffer
        """
        return len(self._frames)
    
    @property
    def dropped_frames(self) -> int:
        """Get count of dropped frames.
        
        Returns:
            int: Number of dropped frames
        """
        return self._dropped_frames


class StreamExtractionError(Exception):
    """Stream extraction error with rich diagnostic context.
    
    Provides categorized error information with original exception tracking
    for intelligent recovery strategies and detailed error reporting.
    
    Attributes:
        original (Optional[Exception]): Original exception that caused this error
        category (str): Error category for programmatic handling
        timestamp (datetime): When the error occurred
    """
    
    def __init__(self, message: str, original: Optional[Exception] = None, category: str = "general") -> None:
        """Initialize with contextual error information.
        
        Args:
            message: Human-readable error description
            original: Original exception that triggered this error
            category: Error category for recovery strategies
        """
        super().__init__(message)
        self.original = original
        self.category = category
        self.timestamp = datetime.now()
    
    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information.
        
        Returns:
            Dict[str, Any]: Structured error details
        """
        return {
            "message": str(self),
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
            "original_type": type(self.original).__name__ if self.original else None,
            "original_message": str(self.original) if self.original else None,
        }


class DependencyError(Exception):
    """Missing dependency error with actionable installation guidance.
    
    Provides clear instructions for resolving dependency issues with
    package details and installation commands.
    
    Attributes:
        package (str): Missing package name
        install_cmd (str): Installation command
        required_for (str): Feature requiring the dependency
    """
    
    def __init__(self, package: str, install_cmd: str, required_for: str = "") -> None:
        """Initialize with package details and installation guidance.
        
        Args:
            package: Missing package name
            install_cmd: Command to install the package
            required_for: Feature requiring this dependency
        """
        feature_info = f" (required for {required_for})" if required_for else ""
        super().__init__(f"Missing dependency: {package}{feature_info}")
        self.package = package
        self.install_cmd = install_cmd
        self.required_for = required_for
    
    def get_installation_instructions(self) -> str:
        """Get user-friendly installation instructions.
        
        Returns:
            str: Formatted installation guidance
        """
        return f"Install {self.package}: {self.install_cmd}"
    

# ╔══════════════════════════════════════════════════════════════╗
# ║ 🌀 Dimensional Transmutation Engine                         ║
# ╚══════════════════════════════════════════════════════════════╝

def image_to_unicode_art(
    pil_image: Image.Image,
    scale_factor: int = 2,
    block_width: int = 8,
    block_height: int = 8,
    edge_threshold: int = 50,
    gradient_str: Optional[str] = None,
    color: bool = True,
    enhanced_edges: bool = True,
    algorithm: str = "sobel",
    dithering: bool = False,
    output_format: str = "ansi",    # Format specification for outputs
    **kwargs: Any                   # Forward compatibility payload
) -> List[str]:
    """🔄 Transform visual matrix into dimensional Unicode with prismatic precision.

    Creates a recursive fractal mapping between visual and typographic spaces,
    compressing multi-dimensional pixel data into a perfectly dense character
    matrix with zero information loss. Each pixel cluster undergoes edge-aware
    transmutation with directional sensitivity and chromatic preservation.
            
    Args:
        pil_image: Source dimensional matrix (PIL Image)
        scale_factor: Detail enhancement coefficient (1-4)
        block_width: Character cell width in pixels
        block_height: Character cell height in pixels
        edge_threshold: Edge detection sensitivity threshold (0-255)
        gradient_str: Character density continuum (None=auto-select)
        color: Enable chromatic dimension mapping
        enhanced_edges: Use directional edge character mapping
        algorithm: Edge detection algorithm variant
        dithering: Apply error diffusion for density precision
        output_format: Output format specifier ("ansi", "plain", "html", "svg")
                
    Returns:
        List[str]: Lines of dimensional Unicode art
    """
    # Select optimal gradient with capability detection
    gradient_str = gradient_str or (
        get_enhanced_gradient_chars() if UNICODE_ENGINE.supports_unicode
        else UNICODE_ENGINE.character_maps["full_gradients"]["ascii_art"]
    )

    # Apply supersampling with optimized scaling
    image_sup = supersample_image(pil_image, scale_factor)
    image_array = np.array(image_sup)
            
    # Apply adaptive image enhancement based on performance tier
    perf_tier = ENV.capabilities["performance_tier"]
    if perf_tier >= 2:
        # High-performance enhancement for detailed processing
        image_array = IMAGE_PROCESSOR.enhance_image(
            image_array, contrast=1.2, brightness=5.0, denoise=True
        )
    elif perf_tier == 1:
        # Moderate enhancement for balanced systems
        image_array = IMAGE_PROCESSOR.enhance_image(
            image_array, contrast=1.1, brightness=0, denoise=False
        )
            
    # Optimize gray and edge processing with vectorized operations
    gray_array = rgb_to_gray(image_array)
    edge_result = IMAGE_PROCESSOR.detect_edges(gray_array, algorithm)
    magnitude, grad_x, grad_y = edge_result["magnitude"], edge_result["gradient_x"], edge_result["gradient_y"]
            
    # Calculate output dimensions with boundary enforcement
    height, width = gray_array.shape
    cols, rows = width // block_width, height // block_height
            
    # Pre-allocate output buffer for optimal memory efficiency
    output_lines = [""] * rows
            
    # Initialize dithering state with perfect calibration
    dither_errors = np.zeros((height, width)) if dithering else None
            
    # Process blocks with optimized loops and vectorized operations
    for i in range(rows):
        # Calculate row slice coordinates with boundary protection
        y_start = i * block_height
        y_end = min((i + 1) * block_height, height)
        line_chars = []
                
        # Process blocks in current row with dimensional awareness
        for j in range(cols):
            # Calculate column slice coordinates with precision
            x_start = j * block_width
            x_end = min((j + 1) * block_width, width)
                    
            # Extract color with efficient slicing and vectorized mean
            color_block = image_array[y_start:y_end, x_start:x_end, :]
            avg_color = np.mean(color_block, axis=(0, 1))
            r, g, b = avg_color.astype(int)
                    
            # Calculate perceptual brightness (ITU-R BT.709 standard)
            brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    
            # Apply dithering with error diffusion for enhanced perception
            if dithering and dither_errors is not None:
                error_avg = np.mean(dither_errors[y_start:y_end, x_start:x_end])
                brightness = np.clip(brightness + error_avg, 0, 255)
                    
            # Detect edges with threshold comparison and statistical analysis
            edge_block = magnitude[y_start:y_end, x_start:x_end]
            avg_edge = np.mean(edge_block)
                    
            # Select character based on edge detection with contextual awareness
            if avg_edge > edge_threshold:
                # Calculate edge direction and strength for directional mapping
                gx = np.mean(grad_x[y_start:y_end, x_start:x_end])
                gy = np.mean(grad_y[y_start:y_end, x_start:x_end])
                edge_strength = min(avg_edge / 255, 1.0)
                        
                # Get directional edge character with optimal parameters
                char = get_edge_char(gx, gy, edge_strength if enhanced_edges else 0.5)
            else:
                # Map brightness to gradient with perfect density mapping
                brightness_norm = brightness / 255
                idx = min(int((1.0 - brightness_norm) * (len(gradient_str) - 1)), 
                          len(gradient_str) - 1)
                char = gradient_str[idx]
                        
                # Apply error diffusion dithering for enhanced visuals
                if dithering and dither_errors is not None:
                    # Calculate quantization error with perceptual precision
                    quantized = (1.0 - idx / (len(gradient_str) - 1)) * 255
                    error = brightness - quantized
                            
                    # Distribute error with Floyd-Steinberg coefficients (optimal pattern)
                    if j < cols - 1:  # Right
                        dither_errors[y_start:y_end, x_end:min(x_end+block_width, width)] += error * 0.4375
                    if i < rows - 1:  # Down
                        dither_errors[y_end:min(y_end+block_height, height), x_start:x_end] += error * 0.3125
                    if i < rows - 1 and j > 0:  # Down-left
                        dither_errors[y_end:min(y_end+block_height, height), 
                                      max(0, x_start-block_width):x_start] += error * 0.1875
                    if i < rows - 1 and j < cols - 1:  # Down-right
                        dither_errors[y_end:min(y_end+block_height, height), 
                                      x_end:min(x_end+block_width, width)] += error * 0.0625
                    
            # Apply color with terminal compatibility awareness
            if color and UNICODE_ENGINE.supports_color:
                # Use optimized ANSI color sequences with perfect boundary checking
                if output_format != "plain":
                    line_chars.append(f"{get_ansi_color(r, g, b)}{char}{reset_ansi()}")
                else:
                    line_chars.append(char)
            else:
                line_chars.append(char)
                
        # Build line with optimized string joining
        output_lines[i] = "".join(line_chars)
            
    return output_lines


def generate_unicode_art(
    image_path: Union[str, Path],
    scale_factor: int = 2,
    block_width: int = 8,
    block_height: int = 8,
    edge_threshold: int = 50,
    gradient_str: Optional[str] = None,
    color: bool = True,
    enhanced_edges: bool = True,
    algorithm: str = "sobel",
    dithering: bool = False,
    auto_scale: bool = True
) -> List[str]:
    """🎭 Process visual matrix into dimensional Unicode with adaptive parameters.
            
    Loads and transmutes an image through a multi-phase transformation pipeline
    with terminal-aware parameter optimization and contextual enhancement.
    Implements precise caching and progress tracking for performance.
            
    Args:
        image_path: Gateway to dimensional source matrix
        scale_factor: Detail enhancement coefficient (1-4)
        block_width: Character cell width in pixels
        block_height: Character cell height in pixels
        edge_threshold: Edge sensitivity threshold (0-255)
        gradient_str: Character density continuum (None=auto-select)
        color: Enable chromatic dimension transfer
        enhanced_edges: Use directional edge character mapping
        algorithm: Edge detection algorithm variant
        dithering: Apply error diffusion for density precision
        auto_scale: Adapt output to terminal dimensions
                
    Returns:
        List[str]: Lines of dimensional Unicode art
                
    Raises:
        FileNotFoundError: If dimensional gateway cannot be accessed
        ValueError: If matrix format is incompatible
    """
    # Start performance tracking with high-precision timer
    start_time = time.perf_counter()
            
    # Apply adaptive parameter optimization based on system capabilities
    if auto_scale:
        params = ENV.get_optimal_params("image")
        terminal_width = ENV.terminal["width"]
        terminal_height = ENV.terminal["height"]
                
        # Optimize scale factor based on system tier
        scale_factor = min(scale_factor, params.get("scale_factor", scale_factor))
                
        # Terminal-aware block sizing with aspect preservation
        if terminal_width < 100:
            block_width = max(block_width, 10)
            block_height = max(block_height, int(block_width * 0.5))
            
    # Load and optimize image with adaptive feedback
    try:
        # Convert image_path to string if it's a Path object to ensure hashability
        image_path_str = str(image_path) if isinstance(image_path, Path) else image_path
        
        if HAS_RICH:
            with CONSOLE.status("📥 Loading dimensional matrix...", spinner="dots"):
                image = Image.open(image_path_str).convert("RGB")
                CONSOLE.log(f"✓ Matrix initialized: {image.width}×{image.height} px")
        else:
            print(f"📥 Initializing dimensional matrix: {image_path_str}")
            image = Image.open(image_path_str).convert("RGB")
            print(f"✓ Matrix initialized: {image.width}×{image.height} px")
    except FileNotFoundError:
        error_msg = f"🚫 Dimensional gateway not found: {image_path_str}"
        if HAS_RICH:
            CONSOLE.print(f"[red bold]{error_msg}[/red bold]")
        else:
            print(error_msg)
        raise
    except Exception as e:
        error_msg = f"🚫 Matrix initialization failed: {e}"
        if HAS_RICH:
            CONSOLE.print(f"[red bold]{error_msg}[/red bold]")
        else:
            print(error_msg)
        raise ValueError(f"Invalid image data: {e}")
            
    # Safety resize for large images with adaptive threshold
    max_dim = 8192 if ENV.capabilities["high_performance"] else 4096
    if image.width > max_dim or image.height > max_dim:
        if HAS_RICH:
            CONSOLE.print(f"[yellow]⚠️ Recalibrating dimensional matrix for stability[/yellow]")
        else:
            print("⚠️ Recalibrating dimensional matrix for stability")
                
        # Preserve aspect ratio during resize
        aspect = image.width / image.height
        if image.width > image.height:
            new_width, new_height = max_dim, int(max_dim / aspect)
        else:
            new_height, new_width = max_dim, int(max_dim * aspect)
                
        # Apply high-quality resize with optimal resampling
        image = image.resize((new_width, new_height), 
                           Image.LANCZOS if ENV.capabilities["high_performance"] else Image.BILINEAR)
            
    # Process image with concurrent execution and progress tracking
    if HAS_RICH:
        with CONSOLE.status("🧠 Transmuting reality through prismatic cipher...", spinner="dots"):
            result = image_to_unicode_art(
                image, scale_factor, block_width, block_height, edge_threshold,
                gradient_str, color, enhanced_edges, algorithm, dithering
            )
            elapsed = time.perf_counter() - start_time
            CONSOLE.log(f"✨ Dimensional transmutation complete in {elapsed:.2f}s")
    else:
        print("🧠 Transmuting reality through prismatic cipher...")
        result = image_to_unicode_art(
            image, scale_factor, block_width, block_height, edge_threshold,
            gradient_str, color, enhanced_edges, algorithm, dithering
        )
        elapsed = time.perf_counter() - start_time
        print(f"✨ Dimensional transmutation complete in {elapsed:.2f}s")
            
    return result


class ArtTransformer:
    """🎨 Dimensional transmutation pipeline with recursive refinement.
    
    Creates self-similar transformation cascades through contextually-aware
    parameter matrices and zero-redundancy rendering pathways. Perfect balance
    of performance and fidelity with structural elegance.
    
    Attributes:
        image (Image.Image): Dimensional source matrix
        options (Dict[str, Any]): Transformation parameter surface
    """
    
    def __init__(self, source: Union[str, Path, Image.Image]) -> None:
        """Initialize transformer with dimensional source matrix.
        
        Args:
            source: Image path, Path object, or PIL Image instance
            
        Raises:
            ValueError: If dimensional matrix cannot be initialized
        """
        # System-aware parameter optimization with zero redundancy
        perf_tier = ENV.capabilities["performance_tier"]
        self.options = {
            "scale_factor": max(1, min(2, perf_tier + 1)),
            "block_width": 8 - (perf_tier * 1),
            "block_height": 8 - (perf_tier * 1),
            "edge_threshold": 50,
            "gradient_str": get_enhanced_gradient_chars() if ENV.capabilities["unicode"] else "@%#*+=-:. ",
            "color": ENV.capabilities["color"],
            "enhanced_edges": perf_tier > 0,
            "algorithm": "scharr" if perf_tier >= 2 else "sobel",
            "dithering": perf_tier >= 2,
            "output_format": "ansi",
            "auto_scale": True
        }
        
        # Dimensional source initialization with perfect error isolation
        try:
            self.image = source if isinstance(source, Image.Image) else Image.open(source).convert("RGB")
        except Exception as e:
            msg = f"Failed to initialize dimensional matrix: {e}"
            if HAS_RICH:
                CONSOLE.print(f"[bold red]🚫 {msg}[/bold red]")
            else:
                print(f"🚫 {msg}")
            raise ValueError(msg) from e
    
    def with_scale(self, factor: int) -> 'ArtTransformer':
        """⚡ Set detail enhancement factor with boundary protection.
        
        Args:
            factor: Detail enhancement multiplier (1-4)
            
        Returns:
            ArtTransformer: Self for method chaining
        """
        self.options["scale_factor"] = max(1, min(4, factor))
        return self
    
    def with_block_size(self, width: int, height: Optional[int] = None) -> 'ArtTransformer':
        """🧩 Set dimensional matrix resolution with aspect preservation.
        
        Args:
            width: Block width in pixels (2+)
            height: Block height in pixels (None=use width)
            
        Returns:
            ArtTransformer: Self for method chaining
        """
        self.options["block_width"] = max(2, width)
        self.options["block_height"] = max(2, height if height is not None else width)
        return self
    
    def with_edge_detection(self, 
                          threshold: int = 50, 
                          algorithm: str = "sobel",
                          enhanced: bool = True) -> 'ArtTransformer':
        """🔍 Configure edge detection with algorithmic precision.
        
        Args:
            threshold: Edge sensitivity threshold (0-255)
            algorithm: Detection algorithm name ('sobel', 'scharr', etc.)
            enhanced: Whether to use directional edge characters
            
        Returns:
            ArtTransformer: Self for method chaining
        """
        self.options["edge_threshold"] = max(0, min(255, threshold))
        self.options["algorithm"] = algorithm
        self.options["enhanced_edges"] = enhanced
        return self
    
    def with_gradient(self, gradient: str) -> 'ArtTransformer':
        """🌈 Set character density gradient with dimensional precision.
        
        Args:
            gradient: Character sequence for density representation
            
        Returns:
            ArtTransformer: Self for method chaining
        """
        if gradient:
            self.options["gradient_str"] = gradient
        return self
    
    def with_preset(self, preset: Literal["default", "detailed", "fast", "minimal"]) -> 'ArtTransformer':
        """⚙️ Apply parameter cascade with contextual optimization.
        
        Args:
            preset: Named parameter configuration
            
        Returns:
            ArtTransformer: Self for method chaining
        """
        # System-aware preset configurations with zero redundancy
        if preset == "detailed":
            self.options.update({
                "scale_factor": min(3, ENV.constraints.get("max_scale_factor", 3)),
                "block_width": 4,
                "block_height": 4,
                "edge_threshold": 40,
                "algorithm": "scharr",
                "enhanced_edges": True,
                "dithering": True
            })
        elif preset == "fast":
            self.options.update({
                "scale_factor": 1,
                "block_width": 12,
                "block_height": 12,
                "edge_threshold": 60,
                "enhanced_edges": False,
                "dithering": False
            })
        elif preset == "minimal":
            self.options.update({
                "scale_factor": 1,
                "block_width": 8,
                "block_height": 8,
                "edge_threshold": 80,
                "color": False,
                "enhanced_edges": False,
                "dithering": False,
                "gradient_str": " .:;+=xX$&#@"
            })
        return self
    
    def with_color(self, enabled: bool = True) -> 'ArtTransformer':
        """🎨 Set color state with capability awareness.
        
        Args:
            enabled: Whether to enable color output
            
        Returns:
            ArtTransformer: Self for method chaining
        """
        self.options["color"] = enabled and ENV.capabilities["color"]
        return self
    
    def with_dithering(self, enabled: bool = True) -> 'ArtTransformer':
        """🔢 Configure error diffusion with perceptual optimization.
        
        Args:
            enabled: Whether to enable dithering
            
        Returns:
            ArtTransformer: Self for method chaining
        """
        self.options["dithering"] = enabled
        return self
    
    def optimize_for_terminal(self) -> 'ArtTransformer':
        """📏 Auto-tune dimensional parameters to terminal constraints.
        
        Calculates optimal block dimensions and scale factor to ensure
        perfect fit within terminal boundaries while preserving aspect
        ratio and structural integrity.
        
        Returns:
            ArtTransformer: Self for method chaining
        """
        # Extract terminal dimensions with fallback
        term_width = ENV.terminal["width"]
        term_height = ENV.terminal["height"]
        
        if self.image and term_width > 10 and term_height > 5:
            # Character aspect ratio compensation (terminal cells are ~2:1)
            char_aspect = 0.5
            
            # Calculate target dimensions with margin preservation
            target_cols = term_width - 4
            target_rows = term_height - 4
            
            # Extract dimensional proportions
            img_width, img_height = self.image.size
            img_aspect = img_width / max(1, img_height)
            
            # Calculate optimal block sizes with aspect preservation
            width_block = max(2, img_width // max(1, target_cols))
            height_block = max(2, img_height // max(1, target_rows))
            
            # Aspect-aware dimensional correction
            adjusted_height = int(width_block / (img_aspect * char_aspect))
            adjusted_width = int(height_block * img_aspect / char_aspect)
            
            # Apply calculated dimensions with boundary constraints
            self.options["block_width"] = min(width_block, adjusted_width)
            self.options["block_height"] = min(height_block, adjusted_height)
            self.options["scale_factor"] = 1 if term_width < 80 or term_height < 24 else self.options["scale_factor"]
            
            # Algorithm optimization for constrained terminals
            if target_cols < 60 or target_rows < 20:
                self.options["algorithm"] = "sobel"  # Faster algorithm for small outputs
                self.options["dithering"] = False    # Disable dithering for small outputs
        
        return self
    
    def render(self) -> List[str]:
        """🔮 Generate dimensional Unicode art with current parameter surface.
        
        Uses concurrent processing for large matrices to maintain
        performance integrity across computational substrates.
        
        Returns:
            List[str]: Lines of dimensional Unicode art
        """
        if not self.image:
            return ["Error: No dimensional matrix initialized"]
        
        # Concurrent rendering for large matrices
        if ENV.capabilities["high_performance"] and self.image.width * self.image.height > 500_000:
            # Thread pool optimization for large matrices
            return THREAD_POOL.submit(
                image_to_unicode_art,
                self.image,
                **self.options
            ).result()
        else:
            # Direct rendering pathway for smaller matrices
            return image_to_unicode_art(
                self.image,
                **self.options
            )
    
    def save_to_file(self, path: Union[str, Path], append_to_collection: bool = True) -> None:
        """💾 Crystallize dimensional art with perfect typographical precision.
        
        Creates format-aware artifacts through contextual transformation paths,
        preserving structural integrity across persistence dimensions. For text
        transmutations, automatically contributes to the central banner collection 
        with multi-font exploration for maximum typographical coverage.
        
        Args:
            path: Target persistence path for dimensional artifact
            append_to_collection: Whether to contribute text banners to collection (default: True)
            
        Raises:
            IOError: If dimensional persistence barrier encountered
        """
        # Format detection matrix with extension mapping
        file_path = Path(path)
        if file_path.suffix.lower() in ('.html', '.htm'):
            self.options["output_format"] = "html" 
        elif file_path.suffix.lower() in ('.svg'):
            self.options["output_format"] = "svg"
        elif file_path.suffix.lower() in ('.ansi'):
            self.options["output_format"] = "ansi"
        
        # For text transformations, always update banner collection unless disabled
        is_text_engine = self.options.get("render_engine") == "text"
        is_text_content = "text_content" in self.options or isinstance(self.image, str)
        
        # Text banner collection integration pathway
        if (is_text_engine or is_text_content) and append_to_collection:
            self._append_to_banner_collection()
            
        # Materialize the dimensional rendering
        result = self.render()
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Format-specific headers with perfect encapsulation
                if self.options.get("output_format") == "html":
                    f.write('<html><head><meta charset="utf-8"><style>'
                           'pre{font-family:monospace;line-height:1.2;background:#111;color:#eee}'
                           'body{margin:0;padding:10px;background:#111}'
                           '</style></head>\n<body><pre>\n')
                elif self.options.get("output_format") == "svg":
                    # Calculate precise dimensions based on content
                    width = max(len(line) for line in result) * 8
                    height = len(result) * 16
                    f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                           f'height="{height}" viewBox="0 0 {width} {height}">\n'
                           f'<rect width="100%" height="100%" fill="#111"/>\n'
                           f'<text x="10" y="20" font-family="monospace" font-size="14" fill="#eee">\n')
                
                # Write content with format-aware transformations
                strip_ansi = self.options.get("output_format") not in ("ansi", "html")
                for i, line in enumerate(result):
                    if strip_ansi:
                        # Extract pure content by removing ANSI codes
                        line = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', line)
                    
                    if self.options.get("output_format") == "svg":
                        # SVG requires tspan elements for proper line positioning
                        f.write(f'<tspan x="10" y="{(i+1)*16}">{line.replace("<", "&lt;").replace("&", "&amp;")}</tspan>\n')
                    else:
                        f.write(line + '\n')
                
                # Format-specific footers with proper structure
                if self.options.get("output_format") == "html":
                    f.write('</pre></body></html>')
                elif self.options.get("output_format") == "svg":
                    f.write('</text>\n</svg>')
            
            # Success notification with minimal frame
            msg = f"✨ Artifact crystallized: {file_path}"
            if HAS_RICH:
                CONSOLE.print(f"[green]{msg}[/green]")
            else:
                print(msg)
                
        except Exception as e:
            # Error with precise contextual framing
            msg = f"Failed to persist dimensional artifact: {e}"
            if HAS_RICH:
                CONSOLE.print(f"[red]🚫 {msg}[/red]")
            else:
                print(f"🚫 {msg}")
            raise IOError(msg) from e

    def _append_to_banner_collection(self) -> None:
        """📚 Forge typographical permutations into universal banner codex.
        
        Creates standardized typographical expressions across multiple font dimensions,
        preserving each in markdown-compatible format with perfect structural harmony.
        Each banner receives precise semantic framing following the exact pattern:
        
        # <Text> Banner
        ```ascii
        <ascii art>
        ```
        
        Uses idempotent operations to eliminate duplicates while ensuring 
        comprehensive font coverage from the pyfiglet universe.
        """
        # Universal collection path - the single source of truth
        collection_path = Path("/home/lloyd/repos/glyph_forge/banner_text.md")
        
        # Extract content with source-aware handling
        text_content = self.options.get("text_content", self.image if isinstance(self.image, str) else "Banner")
        
        # Font exploration strategy - comprehensive typographical coverage
        if "font" in self.options and self.options["font"] is not None:
            # Single font explicitly requested - respect user choice
            fonts = [self.options["font"]]
        else:
            # Comprehensive multi-font exploration - target 20+ variations
            try:
                # First gather all available fonts with perfect retrieval
                all_fonts = TEXT_ENGINE.fonts
                
                # Strategic selection with categorical representation
                if len(all_fonts) <= 25:
                    # Small font universe - include all
                    fonts = all_fonts
                else:
                    # Larger universe - strategic sampling with category integrity
                    categories = TEXT_ENGINE.get_font_categories()
                    selected_fonts = set()
                    
                    # Ensure core fonts are included for baseline representation
                    essential_fonts = ["standard", "slant", "small", "big", "block", "script", 
                                      "digital", "bubble", "mini", "banner", "doom", "isometric1"]
                    for font in essential_fonts:
                        if font in all_fonts:
                            selected_fonts.add(font)
                    
                    # Fill remaining slots with category-aware selection
                    for category in categories:
                        if category == "all":
                            continue
                            
                        category_fonts = TEXT_ENGINE.font_categories.get(category, [])
                        if not category_fonts:
                            continue
                        
                        # Take representative samples from each category
                        sample_size = min(3, max(1, len(category_fonts) // 5))
                        for i in range(sample_size):
                            idx = (i * len(category_fonts) // sample_size)
                            if idx < len(category_fonts):
                                selected_fonts.add(category_fonts[idx])
                    
                    # Finalize selection with count constraint
                    fonts = list(selected_fonts)
                    if len(fonts) < 20:
                        # Supplement with additional fonts to reach target
                        remaining_fonts = [f for f in all_fonts if f not in selected_fonts]
                        step = max(1, len(remaining_fonts) // (25 - len(fonts)))
                        extras = [remaining_fonts[i] for i in range(0, len(remaining_fonts), step)]
                        fonts.extend(extras[:25-len(fonts)])
                
                # Ensure we have at least 20 fonts if available
                if len(fonts) < 20 and len(all_fonts) >= 20:
                    fonts = all_fonts[:25]  # Take top 25 fonts
                    
                # Cap at 25 fonts maximum for performance
                fonts = fonts[:25]
                
            except Exception:
                # Resilient fallback to core fonts
                fonts = ["standard", "slant", "small", "big", "block", "script", "digital", 
                        "bubble", "mini", "banner", "doom", "isometric1", "alligator", 
                        "dotmatrix", "drpepper", "epic", "fuzzy", "ivrit", "lean", "shadow"]
        
        try:
            # Create path structure if needed
            collection_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check for existing entries to ensure idempotency
            existing_content = ""
            existing_headers = set()
            
            if collection_path.exists():
                with open(collection_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                    # Extract existing headers with regex pattern matching
                    for match in re.finditer(r'# ([^\n]+) Banner(?: \(([^\)]+)\))?', existing_content):
                        banner_text = match.group(1)
                        font_name = match.group(2) if match.group(2) else "standard"
                        existing_headers.add(f"{banner_text}:{font_name}")
            
            # Generate new content with duplication prevention
            new_content = []
            fonts_added = 0
            
            for font in fonts:
                # Skip if this exact banner already exists
                header_key = f"{text_content}:{font}"
                if header_key in existing_headers:
                    continue
                    
                # Generate banner with dimensional alignment
                this_banner = text_to_art(
                    text=text_content,
                    font=font,
                    color=None,  # No ANSI in markdown for compatibility
                    width=80,    # Fixed width for collection consistency
                    align=self.options.get("align", "center")
                )
                
                # Skip if generation failed or produced empty result
                if not this_banner or all(not line.strip() for line in this_banner):
                    continue
                
                # Strip any ANSI codes for markdown purity
                this_banner = [re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', line) for line in this_banner]
                
                # Format with perfect markdown structure
                section = []
                section.append(f"# {text_content} Banner ({font})\n")
                section.append("```ascii")
                section.extend(this_banner)
                section.append("```")
                new_content.append("\n".join(section))
                fonts_added += 1
            
            # Only write if new content exists
            if new_content:
                with open(collection_path, 'a', encoding='utf-8') as f:
                    # Add spacing if file already has content
                    if existing_content and not existing_content.endswith('\n\n'):
                        f.write("\n\n")
                    elif existing_content and not existing_content.endswith('\n'):
                        f.write("\n")
                        
                    # Write new banners with spacing
                    f.write("\n\n".join(new_content))
                
                # Success notification with precise metrics
                msg = f"✨ Added {fonts_added} font variations to banner codex ({collection_path.name})"
                if HAS_RICH:
                    CONSOLE.print(f"[green]{msg}[/green]")
                else:
                    print(msg)
            elif fonts_added == 0 and len(fonts) > 0:
                # Skip notification for existing content
                msg = f"ℹ️ All {len(fonts)} font variations already exist in collection"
                if HAS_RICH:
                    CONSOLE.print(f"[blue dim]{msg}[/blue dim]")
                else:
                    print(msg)
                    
        except Exception as e:
            # Error with perfect contextual framing
            msg = f"Failed to update banner collection: {e}"
            if HAS_RICH:
                CONSOLE.print(f"[red]🚫 {msg}[/red]")
            else:
                print(f"🚫 {msg}")
    
    def display(self) -> None:
        """✨ Manifest dimensional art with adaptive presentation.
        
        Automatically calibrates to terminal capabilities with
        enhanced formatting when available. Preserves structural 
        integrity across all display contexts.
        """
        # Generate art with auto-optimization
        if self.options["auto_scale"]:
            self.optimize_for_terminal()
            
        result = self.render()
        
        # Enhanced display with rich when available
        if HAS_RICH:
            # Create presentation panel with metadata
            dims = f"{self.image.width}×{self.image.height} • " if self.image else ""
            quality = f"Scale: {self.options['scale_factor']}× • Algorithm: {self.options['algorithm']}"
            title = f"✨ {dims}{quality} ✨"
            
            # Output with rich formatting
            CONSOLE.print("")
            CONSOLE.print(Panel(title, style="bold blue"))
            for line in result:
                CONSOLE.print(line)
        else:
            # Simple display with separators
            print("\n" + "─" * min(80, ENV.terminal.get("width", 80)) + "\n")
            for line in result:
                print(line)
            print("\n" + "─" * min(80, ENV.terminal.get("width", 80)))


def transform_image(image_path: Union[str, Path], preset: Optional[str] = None) -> List[str]:
    """Transform image into dimensional Unicode art with preset configuration.
    
    Provides a simple one-line interface to the ArtTransformer with optional
    preset selection.
    
    Args:
        image_path: Path to image file
        preset: Optional preset name ("default", "detailed", "fast", "minimal")
        
    Returns:
        List[str]: Lines of dimensional Unicode art
        
    Examples:
        >>> art = transform_image("photo.jpg", "detailed")
        >>> print("\n".join(art))
    """
    transformer = ArtTransformer(image_path)
    
    if preset:
        transformer.with_preset(preset)
    
    # Auto-optimize for terminal display
    return transformer.optimize_for_terminal().render()

# ╔═════════════════════════════════════════════════════════════╗
# ║ 🖼️ Virtual Display Capture And Management                   ║
# ╚═════════════════════════════════════════════════════════════╝

class VirtualDisplayEngine:
    """🔮 Dimensional context bridge for GUI-to-terminal transmutation.
    
    Creates and manages virtual displays (Xvfb), captures their visual state,
    and streams real-time GUI content to terminals through dimensional
    compression with adaptive performance tuning.
    
    Attributes:
        _virtual_display: PyVirtualDisplay instance when active
        _display_process: Direct Xvfb process when using subprocess approach
        _current_display_id: Display number for X11 systems
        _display_size: Resolution tuple (width, height)
        _capabilities: Dict of detected system capabilities
        _capture_backend: Selected optimal screenshot method
        _active_streams: Set of running stream identifiers
    """
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'VirtualDisplayEngine':
        """Get or create the singleton display engine.
        
        Returns:
            VirtualDisplayEngine: Global singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        """Initialize with dynamic capability detection and optimal backends."""
        self._virtual_display = None
        self._display_process = None
        self._current_display_id = 99  # Default virtual display number
        self._display_size = (1280, 720)  # Default resolution
        
        # Core capability detection
        self._capabilities = self._detect_capabilities()
        self._capture_backend = self._select_capture_backend()
        
        # Performance optimization
        self._last_capture_time = 0
        self._active_streams = set()
        self._frame_cache = collections.deque(maxlen=2)

    def _detect_capabilities(self) -> Dict[str, bool]:
        """Detect available virtual display and capture mechanisms.
        
        Returns:
            Dict[str, bool]: Map of capability flags
        """
        capabilities = {
            "has_pyvirtualdisplay": pyvirtualdisplay is not None,
            "has_mss": import_module("mss") is not None,
            "has_pil_grab": hasattr(PIL_Image, "grab") if PIL_Image else False,
            "has_xvfb": shutil.which("xvfb-run") is not None,
            "has_x11_utils": False
        }
        
        # Check for X11 utilities on Linux
        if platform.system() == "Linux":
            capabilities["has_x11_utils"] = any(
                shutil.which(util) for util in ["xwd", "import"]
            )
            
        # Determine if we can create virtual displays
        capabilities["can_create_display"] = (
            capabilities["has_xvfb"] or capabilities["has_pyvirtualdisplay"]
        )
        
        return capabilities

    def _select_capture_backend(self) -> str:
        """Select optimal screenshot backend based on capabilities.
        
        Returns:
            str: Name of best available capture method
        """
        if import_module("mss"):
            return "mss"  # Fast and cross-platform
        elif self._capabilities["has_x11_utils"] and platform.system() == "Linux":
            return "xwd"  # X11 specific but reliable
        elif hasattr(PIL_Image, "grab"):
            return "pillow"  # Decent fallback
        return "none"  # No suitable backend

    def create_virtual_display(self, 
                             width: int = 1280, 
                             height: int = 720, 
                             color_depth: int = 24,
                             visible: bool = False) -> bool:
        """Create new virtual display with specified parameters.
        
        Args:
            width: Display width in pixels
            height: Display height in pixels
            color_depth: Display color depth
            visible: Whether to make display visible (for debugging)
            
        Returns:
            bool: Success status
        """
        if not self._capabilities["can_create_display"]:
            if HAS_RICH:
                CONSOLE.print("[red bold]⚠️ Missing virtual display dependencies[/red bold]")
                CONSOLE.print("[yellow]Install pyvirtualdisplay or xvfb[/yellow]")
            else:
                print("⚠️ Missing virtual display dependencies")
                print("💡 Install pyvirtualdisplay or xvfb")
            return False
            
        # Clean up any existing display
        self.destroy_virtual_display()
        self._display_size = (width, height)
        
        # Try pyvirtualdisplay first (more robust)
        if self._capabilities["has_pyvirtualdisplay"]:
            try:
                self._virtual_display = pyvirtualdisplay.Display(
                    visible=int(visible),
                    size=(width, height),
                    color_depth=color_depth
                )
                self._virtual_display.start()
                os.environ["DISPLAY"] = self._virtual_display.display
                self._current_display_id = int(self._virtual_display.display.replace(':', ''))
                
                if HAS_RICH:
                    CONSOLE.print(f"[green]✓[/green] Virtual display created at [bold]{self._virtual_display.display}[/bold]")
                else:
                    print(f"✓ Virtual display created at {self._virtual_display.display}")
                return True
            except Exception as e:
                if HAS_RICH:
                    CONSOLE.print(f"[red]PyVirtualDisplay error: {e}[/red]")
                else:
                    print(f"PyVirtualDisplay error: {e}")
        
        # Fall back to direct Xvfb if available
        if self._capabilities["has_xvfb"]:
            try:
                # Find available display number
                for display_id in range(99, 110):
                    if not os.path.exists(f"/tmp/.X{display_id}-lock"):
                        self._current_display_id = display_id
                        break
                
                display_str = f":{self._current_display_id}"
                cmd = [
                    "Xvfb", display_str, "-screen", "0", 
                    f"{width}x{height}x{color_depth}", "-nolisten", "tcp"
                ]
                if not visible:
                    cmd.append("-ac")
                    
                self._display_process = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                time.sleep(0.5)  # Brief pause for display initialization
                os.environ["DISPLAY"] = display_str
                
                if HAS_RICH:
                    CONSOLE.print(f"[green]✓[/green] Virtual display created at [bold]{display_str}[/bold]")
                else:
                    print(f"✓ Virtual display created at {display_str}")
                return True
            except Exception as e:
                if HAS_RICH:
                    CONSOLE.print(f"[red]Xvfb error: {e}[/red]")
                else:
                    print(f"Xvfb error: {e}")
                    
        return False

    def destroy_virtual_display(self) -> None:
        """Clean up virtual display resources and active streams."""
        self._active_streams.clear()
        
        if self._virtual_display is not None:
            try:
                self._virtual_display.stop()
            except Exception:
                pass
            self._virtual_display = None
            
        if self._display_process is not None:
            try:
                self._display_process.terminate()
                self._display_process.wait(timeout=1)
            except Exception:
                try:
                    self._display_process.kill()
                except Exception:
                    pass
            self._display_process = None

    def capture_screenshot(self) -> Optional[Image.Image]:
        """Capture screenshot from current display with optimal backend.
        
        Uses the most performant available method with throttling to prevent
        excessive resource usage.
        
        Returns:
            Optional[Image.Image]: PIL Image or None on failure
        """
        # Simple rate limiting to prevent excessive captures
        current_time = time.time()
        if current_time - self._last_capture_time < 0.01:  # Max 100 FPS
            time.sleep(0.01)
        self._last_capture_time = current_time
        
        # Use optimal backend with fallbacks
        if self._capture_backend == "mss":
            try:
                import mss
                with mss.mss() as sct:
                    sct_img = sct.grab(sct.monitors[0])  # Capture entire display
                    return Image.frombytes("RGB", sct_img.size, sct_img.rgb)
            except Exception:
                pass
                    
        elif self._capture_backend == "xwd":
            try:
                display = os.environ.get("DISPLAY", f":{self._current_display_id}")
                process = subprocess.run(
                    ["xwd", "-root", "-display", display, "-silent"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2
                )
                if process.returncode == 0:
                    return Image.open(io.BytesIO(process.stdout))
            except Exception:
                pass
                    
        elif self._capture_backend == "pillow":
            try:
                return PIL_Image.grab()
            except Exception:
                pass
        
        # Final fallback using ImageMagick
        try:
            if shutil.which("import"):
                temp_path = os.path.join(tempfile.gettempdir(), f"glyph_capture_{uuid.uuid4().hex}.png")
                subprocess.run(["import", "-window", "root", temp_path], timeout=2, check=False)
                if os.path.exists(temp_path):
                    img = Image.open(temp_path)
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                    return img
        except Exception:
            pass
            
        return None

    def stream_display_to_terminal(self,
                                 scale_factor: int = 2,
                                 block_width: int = 8,
                                 block_height: int = 8,
                                 edge_threshold: int = 50,
                                 gradient_str: Optional[str] = None,
                                 color: bool = True,
                                 fps: int = 15,
                                 enhanced_edges: bool = True,
                                 max_frames: Optional[int] = None,
                                 algorithm: str = "sobel",
                                 show_stats: bool = True,
                                 adaptive_quality: bool = True) -> None:
        """Stream virtual display content to terminal as Unicode art.
        
        Creates real-time visualization of GUI applications with adaptive quality
        and performance monitoring.
        
        Args:
            scale_factor: Detail enhancement factor (1-4)
            block_width: Character cell width
            block_height: Character cell height
            edge_threshold: Edge detection sensitivity (0-255)
            gradient_str: Custom character gradient (None=auto)
            color: Whether to use ANSI colors
            fps: Target frames per second
            enhanced_edges: Use directional edge characters
            max_frames: Frame capture limit (None=unlimited)
            algorithm: Edge detection algorithm
            show_stats: Display performance metrics
            adaptive_quality: Auto-adjust quality for performance
        """
        # Initialize stream with unique ID and optimal parameters
        stream_id = str(uuid.uuid4())
        self._active_streams.add(stream_id)
        
        # Get system-optimized default parameters
        gradient_str = gradient_str or get_enhanced_gradient_chars()
        frame_interval = 1.0 / fps
        
        # Performance metrics tracking
        metrics = StreamMetrics(sample_size=30)
        
        # Adaptive quality parameters with initial values
        quality_params = RenderParameters(
            scale=scale_factor,
            width=block_width,
            height=block_height,
            threshold=edge_threshold,
            optimal_width=block_width,
            optimal_height=block_height
        )
        thresholds = RenderThresholds.from_target_fps(fps)
        
        # Frame buffer for smoother rendering
        buffer = FrameBuffer(capacity=2)
        
        # Create renderer with terminal-aware parameters
        renderer = FrameRenderer(
            terminal_width=ENV.terminal["width"],
            terminal_height=ENV.terminal["height"],
            gradient=gradient_str,
            border=True, 
            unicode_supported=ENV.capabilities["unicode"]
        )
        
        try:
            # Display stream initialization
            if HAS_RICH:
                CONSOLE.print(Panel(
                    f"🖥️ [bold cyan]Virtual Display Stream[/bold cyan]\n"
                    f"[dim]Resolution: {self._display_size[0]}×{self._display_size[1]} • "
                    f"Target: {fps} FPS • Press Ctrl+C to exit[/dim]",
                    border_style="blue"
                ))
            else:
                print(f"\n🖥️ Virtual Display Stream ({self._display_size[0]}×{self._display_size[1]}) • {fps} FPS")
                print("=" * 50)
                
            # Clear screen before streaming
            renderer.clear_screen()
            
            # Main streaming loop
            frame_count = 0
            while (max_frames is None or frame_count < max_frames) and stream_id in self._active_streams:
                loop_start = time.time()
                
                # Capture screenshot
                screenshot = self.capture_screenshot()
                if screenshot is None:
                    time.sleep(0.1)
                    metrics.record_dropped()
                    continue
                    
                # Process frame with timing
                render_start = time.time()
                art = image_to_unicode_art(
                    screenshot,
                    scale_factor=quality_params.scale,
                    block_width=quality_params.width,
                    block_height=quality_params.height,
                    edge_threshold=quality_params.threshold,
                    gradient_str=gradient_str,
                    color=color,
                    enhanced_edges=enhanced_edges,
                    algorithm=algorithm
                )
                render_time = time.time() - render_start
                
                # Update metrics
                metrics.record_render(render_time)
                metrics.record_frame()
                metrics.update_fps()
                
                # Render complete frame with border and stats
                frame_lines = renderer.render_frame(
                    art=art,
                    source_name=f"Virtual Display ({self._display_size[0]}×{self._display_size[1]})",
                    metrics=metrics,
                    params=quality_params,
                    show_stats=show_stats
                )
                
                # Add to buffer and display
                buffer.add(frame_lines)
                renderer.clear_screen()
                renderer.display_frame(buffer.get_latest())
                
                # Adaptive quality adjustment (every 5 frames)
                if adaptive_quality and frame_count % 5 == 0:
                    quality_params.adjust_quality(render_time * 1000, thresholds)
                
                # Frame rate control
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    metrics.record_dropped()
                
                frame_count += 1
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            if HAS_RICH:
                CONSOLE.print(f"[bold red]🚫 Stream error:[/bold red] {str(e)}")
            else:
                print(f"\n🚫 Stream error: {str(e)}")
        finally:
            # Clean up and show final stats
            if stream_id in self._active_streams:
                self._active_streams.remove(stream_id)
                
            if show_stats and metrics.frames_processed > 0:
                stats = metrics.get_stats()
                if HAS_RICH:
                    CONSOLE.print("\n[bold]📊 Performance summary:[/bold]")
                    stats_table = Table()
                    stats_table.add_column("Metric", style="cyan")
                    stats_table.add_column("Value", style="green")
                    
                    stats_table.add_row("Frames processed", str(stats["total_frames"]))
                    stats_table.add_row("Effective FPS", f"{stats['effective_fps']:.2f}")
                    stats_table.add_row("Render time", f"{stats['avg_render_time']:.1f}ms")
                    stats_table.add_row("Quality level", str(quality_params.quality_level.name))
                    if stats["dropped_frames"] > 0:
                        stats_table.add_row("Dropped frames", 
                                         f"{stats['dropped_frames']} ({stats['drop_ratio']*100:.1f}%)")
                    
                    CONSOLE.print(stats_table)
                else:
                    print("\n📊 Performance summary:")
                    print(f"  • Frames: {stats['total_frames']}")
                    print(f"  • FPS: {stats['effective_fps']:.2f}")
                    print(f"  • Render time: {stats['avg_render_time']:.1f}ms")
                    print(f"  • Quality: {quality_params.quality_level.name}")
                    if stats["dropped_frames"] > 0:
                        print(f"  • Dropped: {stats['dropped_frames']} ({stats['drop_ratio']*100:.1f}%)")

    def launch_gui_application(self, 
                             command: Union[str, List[str]],
                             stream_to_terminal: bool = True,
                             fps: int = 15,
                             timeout: Optional[int] = None,
                             scale_factor: int = 2,
                             color: bool = True) -> subprocess.Popen:
        """Launch GUI application on virtual display with optional streaming.
        
        Args:
            command: Application command or argument list
            stream_to_terminal: Whether to display GUI in terminal
            fps: Frames per second for streaming
            timeout: Maximum runtime in seconds (None=unlimited)
            scale_factor: Detail level for streaming
            color: Enable ANSI color output
            
        Returns:
            subprocess.Popen: Process handle for launched application
            
        Raises:
            RuntimeError: If virtual display cannot be created
        """
        # Ensure virtual display is available
        if not self._virtual_display and self._display_process is None:
            created = self.create_virtual_display()
            if not created:
                raise RuntimeError("Failed to create virtual display")
                
        # Set DISPLAY if not already configured
        if "DISPLAY" not in os.environ:
            os.environ["DISPLAY"] = f":{self._current_display_id}"
            
        # Parse and launch command
        cmd_args = shlex.split(command) if isinstance(command, str) else command
        process = subprocess.Popen(
            cmd_args,
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Stream output if requested
        if stream_to_terminal:
            # Calculate frame limit from timeout if provided
            max_frames = None if timeout is None else int(timeout * fps)
            
            # Allow brief delay for application startup
            time.sleep(0.5)
            
            # Stream the application to terminal
            self.stream_display_to_terminal(
                scale_factor=scale_factor,
                fps=fps,
                color=color,
                max_frames=max_frames,
                adaptive_quality=True,
                show_stats=True
            )
            
        return process

# ╔══════════════════════════════════════════════════════════════╗
# ║ 🌟 Interactive Mode & File Selection                         ║
# ╚══════════════════════════════════════════════════════════════╝

def show_interactive_menu() -> Dict[str, Any]:
    """🎮 Present interactive interface for GlyphStream with adaptive rendering.
    
    Provides contextually-relevant options with multi-stage rendering
    based on system capabilities and user choices. Implements caching and
    concurrent operation preparation for optimal performance.
    
    Returns:
        Dict[str, Any]: User-selected options with validated parameters
    """
    # System-aware defaults with optimized parameters
    sys_params = SYSTEM_CONTEXT.get_optimized_parameters()
    
    # Simplified interface for environments without Rich
    if not HAS_RICH:
        print("\n╔════════ GlyphStream ════════╗")
        print("║ Dimensional Unicode Engine   ║")
        print("╚═══════════════════════════════╝")
        
        source_type = input("Select source (1=Image, 2=Video, 3=YouTube, 4=Webcam, 5=Display): ")
        source = input("Path/URL: ")
        scale = input(f"Detail level (1-4, default={min(2, sys_params.get('scale_factor', 2))}): ")
        color = input("Use color? (y/n, default=y): ")
        
        options = {
            "source": source,
            "scale": int(scale) if scale.isdigit() and 1 <= int(scale) <= 4 else min(2, sys_params.get("scale_factor", 2)),
            "color": color.lower() != 'n',
            "video": source_type in ('2', '3', '4', '5'),
            "virtual_display": source_type == '5',
            "edge_threshold": 50,
            "enhanced_edges": True,
            "algorithm": "sobel",
            "fps": sys_params.get("fps", 15),
            "render_engine": "unicode"
        }
        
        # Intelligent source normalization
        if source_type == '3' and not source.startswith(("http", "www")):
            options["source"] = f"https://youtu.be/{source}"
        elif source_type == '4':
            options["source"] = int(source) if source.isdigit() else 0
            
        print("\n✨ Initializing transmutation engine...\n")
        return options
    
    # Rich-enhanced interface with concurrent resource preparation
    title_art = Text("✨ GLYPH STREAM ✨", style="bold cyan")
    subtitle = Text("Dimensional Unicode Transmutation Engine", style="italic")
    
    # Prepare UI elements in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        header_future = executor.submit(lambda: Panel(
            Group(
                Align.center(title_art),
                Align.center(subtitle),
                Align.center(Text(f"by {AUTHOR_INFO['name']} • {AUTHOR_INFO['org']}", style="dim"))
            ),
            border_style="blue",
            box=rich.box.ROUNDED
        ))
        
        # Pre-load source option table
        source_table_future = executor.submit(lambda: Table(show_header=False, box=rich.box.SIMPLE).add_column("№", style="cyan", no_wrap=True).add_column("Type", style="green", no_wrap=True).add_column("Description", style="white"))
        
        # Concurrently initialize core resources
        if sys_params.get("parallel_decode", True):
            cache_init = executor.submit(UNICODE_ENGINE.get_enhanced_gradient_chars)
    
    # Display header
    CONSOLE.print(header_future.result())
    
    # Source type selection with visual guidance
    source_table = source_table_future.result()
    source_table.add_row("1", "🖼️ [bold]Image[/bold]", "Convert still images into dimensional art")
    source_table.add_row("2", "🎬 [bold]Video[/bold]", "Transform local video files into streams")
    source_table.add_row("3", "📺 [bold]YouTube[/bold]", "Transmute online videos in real-time")
    source_table.add_row("4", "📷 [bold]Webcam[/bold]", "See yourself through the dimensional lens")
    source_table.add_row("5", "🖥️ [bold]Display[/bold]", "Capture virtual display output")
    
    CONSOLE.print(Panel(source_table, title="📦 Source Selection", border_style="cyan"))
    source_type = Prompt.ask("⚡ Choose input type", choices=["1", "2", "3", "4", "5"], default="1")
    
    # Initialize options dict with source type information
    options = {
        "video": source_type in ("2", "3", "4", "5"),
        "virtual_display": source_type == "5"
    }
    
    # Source path/URL with contextual prompting
    if source_type == "1":
        # Image mode
        CONSOLE.print("🖼️ [bold]Image Transmutation Mode[/bold]")
        source = Prompt.ask("📂 Enter image path")
        options["source"] = source
        options["render_engine"] = Prompt.ask(
            "🧪 Select rendering engine",
            choices=["unicode", "text", "transformer"],
            default="unicode"
        )
        
        # Apply preset configuration
        preset = Prompt.ask(
            "🎚️ Select processing mode",
            choices=["standard", "detailed", "fast", "minimal"],
            default="standard"
        )
        
        if preset == "detailed":
            options.update({
                "scale": min(3, sys_params.get("scale_factor", 3)),
                "block_width": 4, "block_height": 4,
                "edge_threshold": 40, "algorithm": "scharr",
                "enhanced_edges": True, "dithering": True
            })
        elif preset == "fast":
            options.update({
                "scale": 1, "block_width": 12, "block_height": 12,
                "edge_threshold": 60, "enhanced_edges": False, "dithering": False
            })
        elif preset == "minimal":
            options.update({
                "scale": 1, "block_width": 8, "block_height": 8,
                "edge_threshold": 80, "enhanced_edges": False, "dithering": False,
                "color": False
            })
    elif source_type == "2":
        # Video mode
        CONSOLE.print("🎬 [bold]Video Stream Mode[/bold]")
        options["source"] = Prompt.ask("📂 Enter video file path")
        options["render_engine"] = "stream"
    elif source_type == "3":
        # YouTube mode
        CONSOLE.print("📺 [bold]YouTube Stream Mode[/bold]")
        youtube_input = Prompt.ask("🔗 Enter YouTube URL or video ID")
        if not youtube_input.startswith(("http", "www")):
            options["source"] = f"https://youtu.be/{youtube_input}"
        else:
            options["source"] = youtube_input
        options["render_engine"] = "stream"
    elif source_type == "4":
        # Webcam mode
        CONSOLE.print("📷 [bold]Realtime Capture Mode[/bold]")
        webcam_id = Prompt.ask("🎯 Enter device ID (usually 0)", default="0")
        options["source"] = int(webcam_id)
        options["render_engine"] = "stream"
    else:
        # Virtual display mode
        CONSOLE.print("🖥️ [bold]Virtual Display Capture Mode[/bold]")
        display_dims = Prompt.ask("📐 Display dimensions (WIDTHxHEIGHT)", default="1280x720")
        try:
            width, height = map(int, display_dims.lower().split('x'))
            options.update({"display_width": width, "display_height": height})
        except (ValueError, TypeError):
            options.update({"display_width": 1280, "display_height": 720})
        options["source"] = 0
        options["render_engine"] = "virtual"
        
        # Check for app launch
        if Confirm.ask("🚀 Launch application in virtual display?", default=False):
            options["launch_application"] = True
            options["application_command"] = Prompt.ask("💻 Enter application command")
    
    # Core parameters with system-aware defaults
    CONSOLE.print(Panel("Configure dimensional transmutation parameters", title="⚙️ Configuration", border_style="cyan"))
    
    # Detail level selection
    default_scale = str(min(2, sys_params.get("scale_factor", 2)))
    options["scale"] = int(Prompt.ask("🔍 Detail level", choices=["1", "2", "3", "4"], default=default_scale))
    
    # Color support with system detection
    options["color"] = Confirm.ask("🎨 Enable dimensional color", default=UNICODE_ENGINE.supports_color)
    
    # Character set selection for unicode engine
    if options.get("render_engine") == "unicode":
        options["gradient_set"] = Prompt.ask(
            "🔠 Select character set",
            choices=["standard", "enhanced", "braille", "ascii"],
            default="standard"
        )
    elif options.get("render_engine") == "text":
        # Font selection for text engine
        font_category = Prompt.ask("🔤 Select font category", default="standard", 
                                  choices=TEXT_ENGINE.get_font_categories())
        options["font"] = TEXT_ENGINE.get_random_font(font_category)
        
    # Video-specific options
    if options["video"]:
        # FPS selection with system-aware defaults
        perf_tier = SYSTEM_CONTEXT.constraints.get("performance_tier", 1)
        default_fps = sys_params.get("default_fps", 15)
        
        options["fps"] = int(Prompt.ask(
            "🎞️ Target frames per second",
            choices=["10", "15", "20", "30"],
            default=str(default_fps)
        ))
        
        # Quality parameters
        options.update({
            "adaptive_quality": Confirm.ask("🧠 Enable adaptive quality", default=True),
            "show_stats": Confirm.ask("📊 Show performance metrics", default=True),
            "border": Confirm.ask("🔲 Add dimensional frame", default=True)
        })
    
    # Advanced options with collapsible interface
    if Confirm.ask("🔬 Configure advanced parameters", default=False):
        # Block size configuration
        options["block_width"] = int(Prompt.ask(
            "⬜ Block width", default=str(sys_params.get("block_width", 8))))
        options["block_height"] = int(Prompt.ask(
            "⬜ Block height", default=str(sys_params.get("block_height", 8))))
        
        # Edge detection parameters
        options["edge_threshold"] = int(Prompt.ask("🔪 Edge threshold", default="50"))
        options["enhanced_edges"] = Confirm.ask("✨ Use enhanced directional edges", 
                                               default=sys_params.get("edge_mode", "enhanced") == "enhanced")
        
        # Algorithm selection for image processing
        if source_type == "1" and options.get("render_engine") == "unicode":
            options["algorithm"] = Prompt.ask(
                "🧪 Edge detection algorithm",
                choices=["sobel", "prewitt", "scharr", "laplacian", "canny"],
                default="sobel"
            )
            options["dithering"] = Confirm.ask("🔢 Apply error diffusion dithering", default=False)
            
        # Text-specific options
        if options.get("render_engine") == "text" and source_type == "1":
            options.update({
                "text_align": Prompt.ask("📏 Text alignment", choices=["left", "center", "right"], default="center"),
                "add_border": Confirm.ask("🔲 Add text border", default=False)
            })
            
        # Transformation pipeline for advanced processing
        if options.get("render_engine") == "transformer" and source_type == "1":
            options["transformations"] = []
            if Confirm.ask("🔧 Add auto-optimization", default=True):
                options["transformations"].append("optimize")
            if Confirm.ask("🔪 Add edge detection", default=True):
                options["transformations"].append("edge")
            if Confirm.ask("🔢 Add dithering", default=False):
                options["transformations"].append("dither")
            if Confirm.ask("🔄 Add color inversion", default=False):
                options["transformations"].append("invert")
    else:
        # Apply intelligent defaults
        options.update({
            "block_width": sys_params.get("block_width", 8),
            "block_height": sys_params.get("block_height", 8),
            "edge_threshold": 50,
            "enhanced_edges": sys_params.get("edge_mode", "enhanced") == "enhanced",
            "algorithm": "sobel",
            "dithering": False
        })
    
    # Save output option
    if Confirm.ask("💾 Save output to file?", default=False):
        options["save_path"] = Prompt.ask("📁 Enter output file path")
        options["output_format"] = Prompt.ask(
            "📄 Select output format",
            choices=["ansi", "plain", "html", "svg", "png"],
            default="ansi"
        )
    
    CONSOLE.print("[bold green]✓[/bold green] Dimensional configuration locked in!")
    CONSOLE.print("[bold blue]🌀 Initializing transmutation engine...[/bold blue]\n")
    
    return options


def parse_command_args() -> Dict[str, Any]:
    """🧩 Parse command-line arguments with intelligent defaults and validation.
    
    Uses system-aware parameter optimization and concurrent resource preparation
    for maximum performance. All options are fully typed and validated with
    contextual auto-correction.
    
    Returns:
        Dict[str, Any]: Parsed and normalized command options
    """
    # Prepare system parameters in parallel with parser setup
    sys_params = SYSTEM_CONTEXT.get_optimized_parameters()
    
    # Create parser with comprehensive options
    parser = argparse.ArgumentParser(
        description="🌟 GlyphStream - Dimensional Unicode Art Transmutation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
╔════════════════════════════════════════════════════╗
║ 🔮 Examples:                                       ║
╟────────────────────────────────────────────────────╢
║ glyph_stream                     # Interactive mode ║
║ glyph_stream image.jpg           # Process image    ║
║ glyph_stream video.mp4 --fps 30  # Process video    ║
║ glyph_stream https://youtu.be/ID # Stream YouTube   ║
║ glyph_stream --webcam            # Use webcam       ║
║ glyph_stream --text "Hello"      # Generate text    ║
╚════════════════════════════════════════════════════╝

{AUTHOR_INFO['name']} <{AUTHOR_INFO['email']}> • {AUTHOR_INFO['org']}
"""
    )
    
    # Source options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('source', nargs='?', help='📂 Input file path or 🔗 URL')
    input_group.add_argument('--webcam', '-w', action='store_true', help='📷 Use webcam as input')
    input_group.add_argument('--virtual-display', '-vd', action='store_true', help='🖥️ Capture virtual display')
    input_group.add_argument('--text', '-tx', help='🔠 Generate text art')
    
    # Core parameters
    parser.add_argument('--engine', '-e', choices=['unicode', 'text', 'transformer', 'stream', 'virtual'],
                       help='🧠 Rendering engine')
    parser.add_argument('--preset', '-p', choices=['standard', 'detailed', 'fast', 'minimal'],
                       help='🧩 Processing preset')
    parser.add_argument('--scale', '-s', type=int, choices=range(1, 5),
                       default=sys_params.get('scale_factor', 2),
                       help='🔍 Detail enhancement factor (1-4)')
    parser.add_argument('--block-width', '-bw', type=int, default=sys_params.get('block_width', 8),
                       help='⬜ Character cell width')
    parser.add_argument('--block-height', '-bh', type=int, default=sys_params.get('block_height', 8),
                       help='⬜ Character cell height')
    parser.add_argument('--edge-threshold', '-et', type=int, default=50,
                       help='🔪 Edge detection threshold (0-255)')
    
    # Feature flags
    parser.add_argument('--no-color', action='store_true', help='⚫ Disable dimensional color')
    parser.add_argument('--no-enhanced-edges', action='store_true', help='➖ Use simplified edge characters')
    parser.add_argument('--dithering', '-d', action='store_true', help='🔢 Apply error diffusion dithering')
    
    # Video options
    parser.add_argument('--fps', '-f', type=int, default=sys_params.get('fps', 15),
                       help='🎞️ Target frames per second for video')
    parser.add_argument('--no-adaptive', action='store_true', help='🔒 Disable adaptive quality')
    parser.add_argument('--no-stats', action='store_true', help='🚫 Hide performance statistics')
    parser.add_argument('--no-border', action='store_true', help='⬜ Disable dimensional frame')
    parser.add_argument('--webcam-id', type=int, default=0, help='🎯 Webcam device ID')
    
    # Advanced options
    parser.add_argument('--algorithm', '-a', 
                       choices=['sobel', 'prewitt', 'scharr', 'laplacian', 'canny'],
                       default='sobel', help='🧪 Edge detection algorithm')
    parser.add_argument('--display-size', help='📐 Virtual display size (WIDTHxHEIGHT)')
    parser.add_argument('--font', help='🔠 Text font name or category')
    parser.add_argument('--align', choices=['left', 'center', 'right'], default='center',
                       help='📏 Text alignment')
    parser.add_argument('--gradient-set', 
                       choices=['standard', 'enhanced', 'braille', 'ascii'],
                       default='standard', help='🔣 Character gradient set')
    parser.add_argument('--transform', '-t', action='append',
                       choices=['optimize', 'edge', 'dither', 'invert'],
                       help='🔄 Apply transformation (repeatable)')
    
    # Output options
    parser.add_argument('--save', '-o', metavar='FILE', help='💾 Save output to file')
    parser.add_argument('--format', choices=['ansi', 'plain', 'html', 'svg', 'png'],
                       default='ansi', help='📄 Output format')
    
    # System options
    parser.add_argument('--debug', action='store_true', help='🐛 Enable debug mode')
    parser.add_argument('--launch-app', help='🚀 Launch application in virtual display')
    parser.add_argument('--benchmark', action='store_true', help='⏱️ Run benchmark')
    
    # Parse args with clean exit handling
    try:
        args = parser.parse_args()
    except SystemExit:
        return {}
    
    # Build options dict with all parameters
    options = {
        'scale': args.scale,
        'block_width': max(1, args.block_width),
        'block_height': max(1, args.block_height),
        'edge_threshold': max(0, min(255, args.edge_threshold)),
        'color': not args.no_color,
        'enhanced_edges': not args.no_enhanced_edges,
        'dithering': args.dithering,
        'algorithm': args.algorithm,
        'fps': max(1, min(60, args.fps)),
        'adaptive_quality': not args.no_adaptive,
        'show_stats': not args.no_stats,
        'border': not args.no_border,
        'gradient_set': args.gradient_set,
        'debug': args.debug,
        'benchmark': args.benchmark,
    }
    
    # Engine selection with auto-detection
    if args.engine:
        options['render_engine'] = args.engine
    
    # Process by source type with smart defaults
    if args.text:
        # Text mode
        options.update({
            'source': args.text,
            'text_content': args.text,
            'font': args.font or 'standard',
            'align': args.align,
            'render_engine': options.get('render_engine', 'text'),
            'video': False
        })
    elif args.webcam:
        # Webcam mode
        options.update({
            'source': args.webcam_id,
            'video': True,
            'render_engine': options.get('render_engine', 'stream')
        })
    elif args.virtual_display:
        # Virtual display mode
        display_width, display_height = 1280, 720
        if args.display_size:
            try:
                parts = args.display_size.lower().split('x')
                if len(parts) == 2:
                    display_width, display_height = map(int, parts)
            except ValueError:
                pass
                
        options.update({
            'source': 0,
            'video': True,
            'virtual_display': True,
            'display_width': display_width,
            'display_height': display_height,
            'render_engine': options.get('render_engine', 'virtual')
        })
        
        # Application launch setup
        if args.launch_app:
            options.update({
                'launch_application': True,
                'application_command': args.launch_app
            })
    elif args.source:
        # Source file/URL
        source = args.source
        
        # Smart YouTube handling
        if any(domain in source.lower() for domain in ['youtube.com', 'youtu.be', 'yt.be']):
            options.update({
                'source': source,
                'video': True,
                'render_engine': options.get('render_engine', 'stream')
            })
        # YouTube ID detection
        elif len(source) == 11 and all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in source):
            options.update({
                'source': f'https://youtu.be/{source}',
                'video': True,
                'render_engine': options.get('render_engine', 'stream')
            })
        else:
            # File detection with MIME-type awareness
            video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv'}
            is_video = False
            
            # Check file extension
            _, ext = os.path.splitext(source.lower())
            if ext in video_extensions:
                is_video = True
                
            options.update({
                'source': source,
                'video': is_video,
                'render_engine': options.get('render_engine', 'stream' if is_video else 'unicode')
            })
    else:
        # No source means interactive mode
        return {}
    
    # Apply quality presets
    if args.preset:
        preset_configs = {
            'detailed': {
                'scale': min(3, sys_params.get('scale_factor', 3)),
                'block_width': 4, 'block_height': 4,
                'edge_threshold': 40, 'algorithm': 'scharr',
                'enhanced_edges': True, 'dithering': True
            },
            'fast': {
                'scale': 1, 'block_width': 12, 'block_height': 12,
                'edge_threshold': 60, 'enhanced_edges': False, 'dithering': False
            },
            'minimal': {
                'scale': 1, 'block_width': 8, 'block_height': 8,
                'edge_threshold': 80, 'color': False, 'enhanced_edges': False,
                'dithering': False, 'gradient_set': 'ascii'
            }
        }
        options.update(preset_configs.get(args.preset, {}))
    
    # Transform pipeline setup
    if args.transform:
        options['transformations'] = args.transform
        # Auto-switch to transformer for pipeline operations
        if options.get('render_engine', 'unicode') == 'unicode':
            options['render_engine'] = 'transformer'
    
    # File output configuration
    if args.save:
        options['save_path'] = args.save
        options['output_format'] = args.format
        
        # Auto-detect format from extension
        if '.' in args.save:
            ext = os.path.splitext(args.save)[1].lower()
            format_map = {
                '.png': 'png', '.jpg': 'png', '.jpeg': 'png',
                '.svg': 'svg', '.html': 'html', '.htm': 'html',
                '.txt': 'plain', '.ansi': 'ansi'
            }
            if ext in format_map:
                options['output_format'] = format_map[ext]
    
    return options


# ╔══════════════════════════════════════════════════════════════╗
# ║ 🚀 Entry Point & Command Interface                          ║
# ╚══════════════════════════════════════════════════════════════╝

def main() -> None:
    """🧠 Dimensional Unicode transmutation orchestration nexus.
    
    Perfect single-point convergence for all transmutation pathways with zero
    redundancy. Every parameter precisely calibrated for maximum perceptual
    density with minimal computational overhead.
    
    Structure:
        1. Command parsing with dimensional intelligence
        2. Source-aware transmutation pipeline selection  
        3. Adaptive parameter optimization based on system context
        4. Concurrent rendering with perfect error isolation
        5. Multi-format persistence with typographical exploration
        
    Returns:
        None: Always returns None with guaranteed clean exit
        
    Raises:
        Nothing: All exceptions contained through perfect error barriers
    """
    # Initialize state early for guaranteed error context
    options = {}
    unicode_art = []
    
    try:
        # Parse arguments or launch interactive menu with perfect fallbacks
        options = parse_command_args() if len(sys.argv) > 1 else show_interactive_menu()
        
        # Exit if help requested or invalid args provided
        if not options:
            return
            
        # Enhanced debug visibility with system traversal
        if options.get("debug", False):
            if HAS_NUMPY:
                np.show_config()
            print(f"🔍 System tier: {ENV.capabilities['performance_tier']}")
            print(f"📐 Terminal: {ENV.terminal['width']}×{ENV.terminal['height']}")
        
        # Typographical transmutation pathway
        if "text" in options or "text_content" in options:
            # Extract text with fallback priority chain
            text_content = options.get("text_content") or options.get("text", "")
            options["text_content"] = text_content
            
            # Create text art with intelligent parameter extraction
            unicode_art = text_to_art(
                text=text_content,
                font=options.get("font", "standard"),
                color=resolve_color(options.get("color_name")) if options.get("color", True) else None,
                width=options.get("max_width", ENV.terminal["width"] - 4),
                align=options.get("align", "center"),
                collection_path="/home/lloyd/repos/glyph_forge/banner_text.md",
                save_multiple_fonts=True,
                font_count=20
            )
            
            # Apply border if requested with zero-overhead styling
            if options.get("add_border", False):
                unicode_art = add_unicode_border(
                    unicode_art, 
                    resolve_color(options.get("color_name")),
                    options.get("border_style", "single")
                )
                
        # Dimensional streaming pathway
        elif options.get("video", False):
            # Stream processing with dimensional optimization
            process_video_stream(
                source=options["source"],
                scale_factor=options["scale"],
                block_width=options.get("block_width", 8),
                block_height=options.get("block_height", 8),
                edge_threshold=options.get("edge_threshold", 50),
                gradient_str=options.get("gradient_str"),
                color=options["color"],
                fps=options.get("fps", 15),
                enhanced_edges=options.get("enhanced_edges", True),
                show_stats=options.get("show_stats", True),
                adaptive_quality=options.get("adaptive_quality", True),
                border=options.get("border", True)
            )
            return  # Early return as stream processing is self-contained
            
        # Advanced transformation pipeline
        elif options.get("render_engine") == "transformer":
            # Initialize transformer with perfect parameter binding
            transformer = ArtTransformer(options["source"])
            
            # Apply transformations with dimensional intelligence
            for transform in options.get("transformations", []):
                if transform == "optimize":
                    transformer.optimize_for_terminal()
                elif transform == "edge":
                    transformer.with_edge_detection(
                        threshold=options.get("edge_threshold", 50),
                        algorithm=options.get("algorithm", "sobel"),
                        enhanced=options.get("enhanced_edges", True)
                    )
                elif transform == "dither":
                    transformer.with_dithering(True)
                elif transform == "invert":
                    # Apply inversion in future updates
                    pass
                    
            # Configure core parameters with zero redundancy
            transformer.with_scale(options["scale"])
            transformer.with_block_size(
                options.get("block_width", 8),
                options.get("block_height", 8)
            )
            transformer.with_color(options["color"])
            
            # Generate art with concurrent resource optimization
            unicode_art = transformer.render()
            
            # Persist typography if this is a text transformation
            if "text_content" in options:
                transformer.options["text_content"] = options["text_content"]
                THREAD_POOL.submit(transformer._append_to_banner_collection)
                
        # Standard image processing pathway
        else:
            # Visual matrix transmutation with contextual parameters
            unicode_art = generate_unicode_art(
                image_path=options["source"],
                scale_factor=options["scale"],
                block_width=options.get("block_width", 8),
                block_height=options.get("block_height", 8),
                edge_threshold=options.get("edge_threshold", 50),
                gradient_str=options.get("gradient_str"),
                color=options["color"],
                enhanced_edges=options.get("enhanced_edges", True),
                algorithm=options.get("algorithm", "sobel"),
                dithering=options.get("dithering", False),
                auto_scale=options.get("auto_scale", True)
            )

        # Terminal output with optimized buffer management
        if unicode_art:
            if HAS_RICH and CONSOLE:
                CONSOLE.print("\n".join(unicode_art))
            else:
                for line in unicode_art:
                    print(line)
                sys.stdout.flush()
        
        # Persistence pathway with format-aware transformation
        if "save_path" in options:
            try:
                # Special handling for text content to ensure banner archival
                if "text_content" in options:
                    # Save to specified path with proper banner formatting
                    save_text_art_to_file(
                        options["text_content"],
                        unicode_art,
                        options["save_path"]
                    )
                    
                    # Always ensure text is archived in collection
                    save_to_banner_collection(
                        options["text_content"],
                        "/home/lloyd/repos/glyph_forge/banner_text.md",
                        True,  # Generate multiple font variants
                        20     # Number of fonts to include
                    )
                else:
                    # Standard file saving for images/other content
                    with open(options["save_path"], 'w', encoding='utf-8') as output_file:
                        # Determine stripping behavior based on format
                        strip_ansi = options.get("output_format") not in ("ansi", "html")
                        ansi_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                        
                        # Format-specific headers with perfect structure
                        if options.get("output_format") == "html":
                            output_file.write('<html><head><meta charset="utf-8">'
                                            '<style>pre{font-family:monospace;line-height:1.2;background:#111;color:#eee}</style>'
                                            '</head>\n<body><pre>\n')
                        
                        # Write content with efficient batching
                        for line in unicode_art:
                            processed_line = ansi_pattern.sub('', line) if strip_ansi else line
                            output_file.write(processed_line + '\n')
                        
                        # Format-specific footers with clean closure
                        if options.get("output_format") == "html":
                            output_file.write('</pre></body></html>')
                
                # Success notification with minimal footprint
                msg = f"✨ Dimensional artifact crystallized: {options['save_path']}"
                if HAS_RICH:
                    CONSOLE.print(f"[green]{msg}[/green]")
                else:
                    print(msg)
                    
            except Exception as e:
                # Error with precise contextual framing
                error_msg = f"🚫 Persistence failure: {str(e)}"
                if HAS_RICH:
                    CONSOLE.print(f"[red]{error_msg}[/red]")
                else:
                    print(error_msg)

    except KeyboardInterrupt:
        # Elegant interruption handling with zero overhead
        print("\n👋 Dimensional transmutation gracefully aborted", flush=True)
    except Exception as e:
        # Enhanced error handling with perfect recovery paths
        error_type = type(e).__name__
        error_msg = str(e) or "Unknown dimensional anomaly"
        
        # 🛡️ Safe error reporting with nested resilience
        try:
            if HAS_RICH and CONSOLE:
                CONSOLE.print(f"\n[bold red]🚫 {error_type}:[/bold red] {error_msg}")
                if options.get("debug", False) and hasattr(e, "__traceback__"):
                    CONSOLE.print_exception()
                CONSOLE.print("[yellow]💡 For troubleshooting, run with --help or in interactive mode[/yellow]")
            else:
                print(f"\n🚫 {error_type}: {error_msg}")
                print("💡 Run with --help for usage information")
        except Exception:
            # 🔄 Ultimate fallback for critical failures
            print(f"\n🚫 Error: {error_type}: {error_msg}")
            if options.get("debug", False):
                traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Initialize dimensional transmutation matrix
    main()
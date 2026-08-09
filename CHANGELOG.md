# 📜 CHANGELOG

> *"History is a linked list of intentional transformations."*

## ⚡ Glyph Forge Version Timeline ⚡

This document maps the evolution pathway of Glyph Forge through its structural iterations. Each release represents a precise advancement with measurable impact.

<!-- 
    Change classification system:
    ⚠️ Breaking - Interface restructuring
    ✨ Features - Capability expansion
    🐛 Fixes - Error resolution
    ⚡ Performance - Execution optimization
    📚 Documentation - Knowledge crystallization
    🔧 Refactor - Internal restructuring
    🔒 Security - Protection enhancements
 -->

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## [0.1.0] - 2024-11-15

> *"Initial implementation—where pixels find their glyph essence."*

### ✨ Added

- Core transformation engine with edge-preserving algorithms
- Multi-format renderer ecosystem (Text, ANSI, HTML, SVG)
- Terminal-aware color system with fallback pathways
- Image processing pipeline with density mapping
- CLI entry points: `imagize` (alias: `glyphfy`) and `bannerize`
- Comprehensive type annotations across all interfaces
- Documentation system with practical examples

### 🔧 Technical Implementation

- Character density mappings with contextual boundary detection
- Gradient-preserving transformation with minimal information loss
- Color representation system with environment adaptation
- Performance baseline established (0.09s standard processing time)
- Verification system with edge case coverage

### 📚 Documentation

- Installation guide with dependency explanation
- Quick start examples for immediate productivity
- API reference with precise parameter definitions
- Architectural overview with component relationships
- Contribution guidelines with workflow specifications

### 🛠️ Infrastructure

- Continuous integration pipeline with validation gates
- Style enforcement hooks for pre-commit verification
- Distribution system through PyPI with integrity checks
- Development environment configuration

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## [Unreleased]

> *"Functionality expands like glyph itself—simple elements creating complex structures."*

### ✨ Added

- Unified `glyph-forge video` command with adaptive eco, balanced, and
  workstation profiles
- Full-colour vectorized glyph-atlas video renderer with streamed FFmpeg output
- Source-audio muxing, subclip controls, encoder quality controls, progress
  callbacks, portable font discovery, and atomic output saves
- Low-latency glyph, Braille 2×4, true-color half-block, and quadrant renderers
- Scalable SVG text export for lossless zoomable stills
- `glyph-forge live camera`, `live screen`, and `live video` commands plus the
  convenient `webcam` and `desktop` aliases
- Cross-platform OpenCV camera/video capture and MSS/Pillow screen capture
  behind a shared source protocol
- Private local browser Studio with image/video drag and drop, webcam and screen
  capture, WebGL2 glyph-atlas previews, Canvas2D fallback, live controls, and
  PNG/SVG/TXT export
- Browser-native sharing and copyable style links without uploading private
  media to a Glyph Forge service
- A redesigned full-screen TUI with filtered media browsing, image and text
  previews, live sources, saving, runtime diagnostics, and Studio handoff
- Executable self-checks in `glyph-forge doctor` so broken FFmpeg installations
  are reported accurately instead of being treated as available

### ⚡ Performance

- Video frames now stream directly from decoder to renderer to encoder without
  temporary images or unbounded frame lists
- Existing video frame APIs now use lazy iterators internally while preserving
  their list-returning compatibility functions
- Live capture uses a single newest-frame slot and intentionally drops stale
  frames when rendering falls behind, keeping latency and memory bounded
- The browser Studio renders through a GPU glyph atlas when WebGL2 is available
  and automatically falls back to a portable Canvas2D renderer

### 🔒 Security

- Browser Studio binds to loopback by default and requires an explicit
  `--allow-network` flag before accepting a non-loopback address
- Studio responses include a restrictive content-security policy, disable
  caching, and prevent MIME sniffing, framing, and referrer leakage

### 🔧 Repository

- Consolidated development history onto `main`, made it the GitHub default,
  and removed redundant local and remote branches
- Integrated the standalone glyph-video prototype into the tested package and
  unified CLI, eliminating the duplicate script without losing its controls

### 🔮 Development Vector

- Format-specific rendering optimizations
- Pattern recognition system with feature preservation
- Extended format support with conversion integrity
- Optional desktop input routing with explicit permissions and focus control
- Stable API contracts with backward compatibility
- CLI enhancement with progress visualization

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

© 2023-2024 [Lloyd Handyside](mailto:ace1928@gmail.com) & [Eidos](mailto:syntheticeidos@gmail.com) — Maintained by [Neuroforge](https://neuroforge.io).

"A changelog is like glyph art—structured information that tells a complete story."

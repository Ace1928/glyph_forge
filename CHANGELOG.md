# Changelog

Notable user-visible changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

No changes yet.

## [1.0.0] - 2026-07-13

### Added

- Standalone package, CLI, documentation, tests, and release automation.
- Image conversion to plain text, ANSI 16/256, truecolor, and HTML.
- Adaptive streaming renderer with gradient, blocks, Braille, and ASCII modes.
- Video, webcam, URL, screen-capture, recording, audio, gallery, and TUI feature
  stacks behind optional dependencies.
- Side-effect-free terminal capability detection and machine-readable doctor
  output.
- Layered system, user, and runtime configuration with atomic persistence.
- Stable package exception hierarchy and conventional `image_to_glyph` API.
- Cross-platform CI, wheel smoke tests, dependency audit, completions, and
  contributor/security policies.

### Changed

- Package import is now lazy and free of logging, filesystem, signal, and
  optional dependency side effects.
- Streaming defaults adapt to TTY state, color depth, encoding, `NO_COLOR`, and
  `TERM=dumb`.
- Galleries consume caller-provided media instead of a repository-managed asset
  library.
- The banner command is directly available as `glyph-forge bannerize TEXT`.
- Configuration now keeps runtime values out of the user file and treats system
  configuration as read-only.

### Security

- Replaced pickle-based streaming caches with validated `allow_pickle=False`
  NPZ archives and atomic owner-only writes.
- Added early frame validation, escaped gallery HTML, safer output-directory
  behavior, and explicit confirmation for sensitive capture/network operations.

### Deprecated

- `GlyphForgeAPI.image_to_Glyph`; use `image_to_glyph`. The alias remains for
  the 1.x line.

## [0.1.0] - 2024-11-15

### Added

- Initial banner, image transformation, renderer, and command-line modules.

[Unreleased]: https://github.com/Ace1928/glyph_forge/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Ace1928/glyph_forge/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/Ace1928/glyph_forge/releases/tag/v0.1.0

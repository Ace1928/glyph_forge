# 📋 Glyph Forge TODO

> Roadmap and planned improvements for the Glyph Forge project.

## 🚀 Version 0.2.0 (Next Release)

### High Priority

- [x] **Implement actual transformer classes** - Replace placeholder implementations in `transformers/__init__.py`
  - [x] `ImageTransformer` - Full image-to-matrix transformation
  - [x] `ColorMapper` - Color space conversions and mapping
  - [x] `DepthAnalyzer` - Depth-based character selection
  - [x] `EdgeDetector` - Edge detection for enhanced detail

- [ ] **Add video processing with OpenCV** - Full video support beyond GIFs
  - [ ] Video file decoding (MP4, AVI, MKV)
  - [ ] Frame rate control
  - [ ] Audio synchronization markers

- [ ] **Improve CLI error handling**
  - [ ] Better error messages for missing files
  - [ ] Validation of parameter combinations
  - [ ] Graceful handling of unsupported formats

### Medium Priority

- [ ] **Add more dithering algorithms**
  - [ ] Ordered (Bayer matrix) dithering
  - [ ] Blue noise dithering
  - [ ] Pattern dithering

- [ ] **Enhance color mode support**
  - [ ] 8-color ANSI mode
  - [ ] Web-safe color palette
  - [ ] Custom color palettes

- [ ] **Performance optimizations**
  - [ ] SIMD optimizations for pixel processing
  - [ ] GPU acceleration (optional)
  - [ ] Streaming mode for large files

- [ ] **Documentation improvements**
  - [ ] Sphinx API documentation
  - [ ] Interactive examples
  - [ ] Video tutorials

### Low Priority

- [ ] **Add new character sets**
  - [ ] Emoji-based density mapping
  - [ ] Custom font loading
  - [ ] Unicode normalization options

- [ ] **TUI enhancements** (textual-based)
  - [ ] Real-time preview
  - [ ] Parameter adjustment sliders
  - [ ] History/undo support

## 🔧 Version 0.3.0 (Future)

### Planned Features

- [ ] **Plugin system** - Allow third-party extensions
  - [ ] Custom renderer plugins
  - [ ] Custom transformer plugins
  - [ ] Character set plugins

- [ ] **Web API** - RESTful API for web integration
  - [ ] Flask/FastAPI server
  - [ ] WebSocket streaming
  - [ ] Docker container

- [ ] **Machine learning integration**
  - [ ] Neural style transfer
  - [ ] Object detection for smart cropping
  - [ ] Semantic segmentation for region-based styling

## 🐛 Known Issues

### Critical
- None currently

### High
- [ ] Large images (>8K) may cause memory issues
- [ ] Some FIGlet fonts have rendering artifacts

### Medium
- [ ] Terminal auto-sizing may not work in all environments
- [ ] Windows console color support is limited
- [ ] Some Unicode characters may not render in all terminals

### Low
- [ ] Cache cleanup may leave orphan entries under high load
- [ ] Documentation links need updating after restructure

## 📈 Metrics Goals

- [ ] Test coverage: 85% → 95%
- [ ] Documentation coverage: 70% → 100%
- [ ] Performance: 10% improvement in image conversion
- [ ] Memory: 20% reduction in peak memory usage

## 🔒 Security Enhancements

- [ ] Add input sanitization for file paths
- [ ] Implement rate limiting in API
- [ ] Add content validation for uploaded images
- [ ] Security audit of dependencies

## 📚 Documentation Tasks

- [x] Comprehensive README.md
- [x] TODO.md (this file)
- [x] CURRENT_STATUS.md
- [ ] API reference documentation
- [ ] Architecture decision records (ADRs)
- [ ] Performance benchmarking guide
- [ ] Deployment guide

## 🧪 Testing Tasks

- [x] Unit tests for core modules
- [x] Integration tests for API
- [ ] End-to-end CLI tests
- [ ] Performance regression tests
- [ ] Cross-platform tests
- [ ] Memory leak tests

## 💡 Ideas for Consideration

These are ideas that need further evaluation:

1. **Live terminal preview mode** - Real-time conversion with webcam input
2. **ASCII art editor** - Interactive editing of generated art
3. **Animation export** - Export sequences as animated GIFs
4. **Diff visualization** - Show image differences in ASCII
5. **QR code support** - Generate ASCII QR codes
6. **Steganography** - Hide data in ASCII art
7. **3D rendering** - ASCII art from 3D models

---

## How to Contribute

1. Pick an item from the TODO list
2. Create a GitHub issue to track the work
3. Fork the repository
4. Create a feature branch
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

*Last updated: January 2026*

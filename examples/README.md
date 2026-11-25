# Examples

`api_examples.py` uses only the supported high-level API. It renders a banner
without external assets and optionally converts an image you provide.

```bash
python examples/api_examples.py --help
python examples/api_examples.py
python examples/api_examples.py ./photo.jpg --width 96 --color
```

Run it from an activated environment where Glyph Forge is installed. Failures
use concise messages and nonzero exit codes.

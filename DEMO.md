# Demo recipes

## No assets or optional dependencies

```bash
glyph-forge demo
glyph-forge demo --mode banner
glyph-forge bannerize "GLYPH FORGE" --font slant --style boxed
```

The image demo generates a gradient in memory and removes its temporary source
file automatically. It performs no network request.

## Image

```bash
glyph-forge imagize convert ./photo.jpg --width 100 --color none
glyph-forge imagize convert ./photo.jpg \
  --width 120 \
  --color ansi256 \
  --dither \
  --dither-algorithm atkinson \
  --edge-enhance
```

## Gallery

```bash
glyph-forge gallery ./photos \
  --output-dir ./glyph-gallery \
  --formats html,svg,png \
  --preset cinematic
```

## Video and webcam

Install the optional streaming stack first:

```bash
python -m pip install ".[streaming]"
glyph-forge doctor
glyph-forge stream ./clip.mp4 --mode auto --color auto --no-audio
glyph-forge stream --webcam 0 --resolution 480p --fps 24
```

For a URL, pass a source you are permitted to access and answer the network
confirmation prompt. Glyph Forge does not bundle third-party media or bypass
service access controls.

## Portable capture for logs

```bash
NO_COLOR=1 GLYPH_FORGE_ASCII=1 \
  glyph-forge stream ./clip.mp4 --mode auto --color auto --no-audio > frames.txt
```

Redirected streaming writes sequential newline-separated frames without
clear-screen or cursor-home codes.

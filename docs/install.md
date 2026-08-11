# Install and launch Glyph Forge

Choose the route that matches how you want to use Glyph Forge:

| Route | Python needed | Updates | Best for |
|---|---:|---|---|
| browser / installed web app | no | automatic | phones, tablets, Chromebooks, and instant sharing |
| `uv tool` | managed automatically | reinstall the next versioned tag | fastest isolated install |
| pipx | managed automatically | reinstall the next versioned tag | familiar isolated CLI install |
| portable archive | no | replace the unpacked directory | friends and machines without Python |
| `pip` in a venv | yes, 3.10–3.14 | `pip install --upgrade` | Python/API development |

Install the Python distribution named `glyphforge` (no separator). It provides
the `glyph-forge` command and `glyph_forge` import. The hyphenated PyPI project
`glyph-forge` is an unrelated API client and is not this repository. Until the
available `glyphforge` name receives its one-time PyPI owner configuration, the
immutable `v0.3.1` GitHub tag below is the canonical Python source.

## Install the browser Studio

Open [ace1928.github.io/glyph_forge](https://ace1928.github.io/glyph_forge/).
Chrome and Edge expose an **Install app** prompt; Safari uses **Add to Dock** on
macOS or **Add to Home Screen** on iPhone and iPad. The installed Studio runs in
its own window and can reopen its editor shell offline after the first visit.
Selected images, video, camera/screen streams, exports, API responses, and
temporary shares are excluded from offline storage.

The web route needs no Python and remains useful when a browser does not offer
installation. Camera, microphone, screen capture, recording, file-handler, and
Web Share support follow the browser and operating system permission model;
controls whose underlying API is unavailable are disabled without affecting
the rest of the editor.

Studio separates final output pixels from glyph sampling density. Presets span
adaptive through 8K, exact width/height fields support aspect locking, and the
same selected size drives PNG, SVG, sharing, and recording. Requests beyond a
device's reported canvas/GPU limit are fitted proportionally and shown in the
controls instead of failing during export.

## Fast isolated install

With [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv tool install "glyphforge[all] @ https://github.com/Ace1928/glyph_forge/archive/refs/tags/v0.3.1.zip"
glyph-forge launch
```

Try it once without keeping an environment:

```bash
uvx --from "glyphforge[all] @ https://github.com/Ace1928/glyph_forge/archive/refs/tags/v0.3.1.zip" glyph-forge launch
```

With [pipx](https://pipx.pypa.io/stable/installation/):

```bash
pipx install "glyphforge[all] @ https://github.com/Ace1928/glyph_forge/archive/refs/tags/v0.3.1.zip"
glyph-forge launch
```

Both tools isolate Glyph Forge from system Python packages while exposing its
commands on the user path. Use `glyph-forge doctor` immediately after install
to see exactly which cameras, displays, media tools, and interfaces are ready.

## Portable app archives

Tagged GitHub releases build and smoke-test a one-directory app on current
Windows, macOS, and Linux runners. Download the archive matching the OS and
processor, unpack the whole directory, then run:

```text
Windows: glyph-forge.exe launch
macOS:   ./glyph-forge launch
Linux:   ./glyph-forge launch
```

Do not move only the executable: its `_internal` directory contains Python,
native libraries, styles, and the browser Studio. The one-directory layout is
intentional—it starts immediately instead of unpacking a large temporary app on
every run. Add the directory to `PATH` or create a normal shortcut if desired.

Release bundles include the TUI, OpenCV/MSS media capture, yt-dlp URL support,
virtual-display integration, and input-control Python adapters. Platform tools
such as FFmpeg/ffprobe and Xvfb are not silently redistributed; install them
through the operating system when `glyph-forge doctor` requests them. Webcam,
screen, and input access still follow OS permission prompts.

## Verify a release

Every release contains `SHA256SUMS`. On Linux/macOS, from the download folder:

```bash
sha256sum --check SHA256SUMS
```

On PowerShell:

```powershell
Get-FileHash .\glyph-forge-*.zip -Algorithm SHA256
Get-Content .\SHA256SUMS
```

GitHub also signs build provenance through Sigstore. With GitHub CLI:

```bash
gh attestation verify glyph-forge-VERSION-PLATFORM.ARCHIVE \
  --repo Ace1928/glyph_forge
```

The attestation binds an asset digest to this repository and release workflow.
It is supply-chain provenance, not Windows Authenticode or Apple notarization;
the latter require private platform certificates and are never implied.

## Small or custom installs

The core is enough for images, text, Studio, demos, and diagnostics:

```bash
python -m venv .venv
python -m pip install "glyphforge @ https://github.com/Ace1928/glyph_forge/archive/refs/tags/v0.3.1.zip"
glyph-forge studio
```

Install focused extras rather than everything when disk or memory is tight:

```bash
python -m pip install "glyphforge[tui] @ https://github.com/Ace1928/glyph_forge/archive/refs/tags/v0.3.1.zip"
python -m pip install "glyphforge[media] @ https://github.com/Ace1928/glyph_forge/archive/refs/tags/v0.3.1.zip"
python -m pip install "glyphforge[network] @ https://github.com/Ace1928/glyph_forge/archive/refs/tags/v0.3.1.zip"
```

The runtime profiles (`eco`, `balanced`, and `workstation`) are independent of
installation size. `--performance auto` remains the recommended default.

## Upgrade or remove

```bash
uv tool install --force "glyphforge[all] @ https://github.com/Ace1928/glyph_forge/archive/refs/tags/v0.3.1.zip"
uv tool uninstall glyphforge

pipx install --force "glyphforge[all] @ https://github.com/Ace1928/glyph_forge/archive/refs/tags/v0.3.1.zip"
pipx uninstall glyphforge
```

For a portable archive, stop Glyph Forge and replace the entire unpacked
directory. User configuration remains under the platform's normal home/config
location and is not stored inside the app directory.

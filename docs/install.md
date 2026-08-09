# Install and launch Glyph Forge

Choose the route that matches how you want to use Glyph Forge:

| Route | Python needed | Updates | Best for |
|---|---:|---|---|
| `uv tool` | managed automatically | `uv tool upgrade glyph-forge` | fastest isolated install |
| pipx | managed automatically | `pipx upgrade glyph-forge` | familiar isolated CLI install |
| portable archive | no | replace the unpacked directory | friends and machines without Python |
| `pip` in a venv | yes, 3.10–3.14 | `pip install --upgrade` | Python/API development |

## Fast isolated install

With [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv tool install "glyph-forge[all]"
glyph-forge launch
```

Try it once without keeping an environment:

```bash
uvx --from "glyph-forge[all]" glyph-forge launch
```

With [pipx](https://pipx.pypa.io/stable/installation/):

```bash
pipx install "glyph-forge[all]"
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
python -m pip install glyph-forge
glyph-forge studio
```

Install focused extras rather than everything when disk or memory is tight:

```bash
python -m pip install "glyph-forge[tui]"
python -m pip install "glyph-forge[media]"
python -m pip install "glyph-forge[network]"
```

The runtime profiles (`eco`, `balanced`, and `workstation`) are independent of
installation size. `--performance auto` remains the recommended default.

## Upgrade or remove

```bash
uv tool upgrade glyph-forge
uv tool uninstall glyph-forge

pipx upgrade glyph-forge
pipx uninstall glyph-forge
```

For a portable archive, stop Glyph Forge and replace the entire unpacked
directory. User configuration remains under the platform's normal home/config
location and is not stored inside the app directory.

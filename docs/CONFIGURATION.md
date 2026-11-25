# Configuration

Glyph Forge configuration has deterministic precedence:

```text
built-in defaults < system file < user file < runtime overrides
```

## Locations

On Linux and other XDG environments:

- System: `/etc/glyph-forge/config.json`
- User: `$XDG_CONFIG_HOME/glyph-forge/config.json`
- User fallback: `~/.config/glyph-forge/config.json`

On Windows, the system and user files live below `PROGRAMDATA` and `APPDATA`
respectively in a `Glyph Forge` directory.

Environment overrides:

- `GLYPH_FORGE_SYSTEM_CONFIG`: explicit system configuration file.
- `GLYPH_FORGE_CONFIG_FILE`: explicit user configuration file.
- `GLYPH_FORGE_CACHE_DIR`: streaming lookup-table cache directory.

Reads never create paths. A user directory is created only after an explicit
persistent setting change.

## File format

Files are UTF-8 JSON objects grouped by section:

```json
{
  "banner": {
    "default_font": "slant",
    "default_width": 100
  },
  "image": {
    "default_charset": "general",
    "default_width": 96,
    "dithering": true
  }
}
```

Invalid or unreadable configuration is ignored with a warning so the CLI can
start with safe defaults. Programmatic writes reject non-JSON values and
non-finite numbers.

## Python API

```python
from glyph_forge.config import ConfigScope, get_config

config = get_config()
width = config.get("image", "default_width", 80)

# Persist only this user override.
config.set("image", "default_width", 96)

# Apply an in-memory override that is never serialized.
config.set("image", "default_width", 120, scope=ConfigScope.RUNTIME)
```

System configuration is read-only inside Glyph Forge. Administrators should
manage it with their normal deployment tooling. User writes use a temporary
file, flush, `fsync`, atomic replacement, and owner-only POSIX permissions.

`reset_to_defaults()` removes user and runtime overrides. A section name may be
passed to reset only that section. System settings remain in effect.

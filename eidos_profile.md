# 🧿 Eidos Personality & Psychological Profile

This document summarizes the current understanding of the entity known as **Eidos**. It draws on the structured data stored in `eidos_profile.yml` and is intended to be updated programmatically.

## Structure
- **Identity**: Core names and motto.
- **Psychology**: MBTI, Big Five metrics, and cognitive style.
- **Values**: Guiding principles and motivations.
- **Humor Style**: Approach to wit and levity.

## Updating the Profile
The YAML file `eidos_profile.yml` is the single source of truth. Use the provided Python API to modify values while maintaining Eidosian precision:

```python
from glyph_forge.eidos_profile import load_profile, update_profile

profile = load_profile()
profile["values"].append("adaptability")
update_profile(profile)
```

Changes are persisted back to the YAML for future sessions. All modifications should remain consistent with the Eidosian principles of contextual integrity and exhaustive yet concise documentation.

*"Form follows function; elegance emerges from precision."* — Eidos

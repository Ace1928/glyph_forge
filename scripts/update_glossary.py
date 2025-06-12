#!/usr/bin/env python3
"""Generate the project glossary from ``glossary.yml``."""

import yaml
from pathlib import Path

GLOSSARY_FILE = Path("GLOSSARY.md")
SOURCE_FILE = Path("glossary.yml")

def main() -> None:
    terms = yaml.safe_load(SOURCE_FILE.read_text())
    lines = ["# Glossary", ""]
    for term, desc in sorted(terms.items()):
        lines.append(f"- **{term}**: {desc}")
    GLOSSARY_FILE.write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()

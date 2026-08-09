"""Compatibility entry point for tooling that still invokes ``setup.py``.

All project metadata lives in ``pyproject.toml``.  Keeping this tiny bridge
avoids two sources of truth while remaining friendly to older installers.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()

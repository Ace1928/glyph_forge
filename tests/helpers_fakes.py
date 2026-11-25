"""Test fakes for external dependencies."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


class DummyCompletedProcess:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class DummyPopen:
    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs
        self.pid = 12345
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO()
        self.stderr = io.BytesIO()
        self._returncode = None

    @property
    def returncode(self):
        return self._returncode

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def communicate(self, input: Optional[bytes] = None, timeout: Optional[float] = None):
        if input is not None:
            self.stdin.write(input)
        self._returncode = 0
        return self.stdout.getvalue(), self.stderr.getvalue()

    def poll(self):
        return self._returncode

    def wait(self, timeout: Optional[float] = None):
        self._returncode = 0
        return self._returncode

    def terminate(self):
        self._returncode = -15

    def kill(self):
        self._returncode = -9


class DummyVideoCapture:
    def __init__(self, source: Any):
        self.source = source
        self._opened = True
        self._frame = np.zeros((10, 10, 3), dtype=np.uint8)
        self._pos = 0

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if self._pos > 0:
            return False, None
        self._pos += 1
        return True, self._frame.copy()

    def release(self) -> None:
        self._opened = False

    def set(self, prop: int, value: Any) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == 5:  # CAP_PROP_FPS
            return 30.0
        if prop == 3:  # CAP_PROP_FRAME_WIDTH
            return 10.0
        if prop == 4:  # CAP_PROP_FRAME_HEIGHT
            return 10.0
        if prop == 7:  # CAP_PROP_FRAME_COUNT
            return 1.0
        return 0.0


@dataclass
class DummyMSSMonitor:
    left: int = 0
    top: int = 0
    width: int = 10
    height: int = 10

    def __getitem__(self, item: str):
        return getattr(self, item)


class DummyMSS:
    def __init__(self):
        self.monitors = [DummyMSSMonitor(), DummyMSSMonitor(width=8, height=8)]

    def grab(self, monitor: Any):
        w = monitor["width"]
        h = monitor["height"]
        # BGRA
        return np.zeros((h, w, 4), dtype=np.uint8)

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class DummyPlaywrightPage:
    def __init__(self):
        self._selectors = {}

    def goto(self, url: str, timeout: int = 60000):
        return None

    def wait_for_selector(self, selector: str, timeout: int = 1000):
        return True

    def query_selector(self, selector: str):
        return None

    def screenshot(self, type: str = "png"):
        return b"\x89PNG\r\n\x1a\n"  # minimal PNG header


class DummyPlaywrightContext:
    def __init__(self):
        self.pages = [DummyPlaywrightPage()]

    def new_page(self):
        return DummyPlaywrightPage()


class DummyPlaywrightBrowser:
    def new_context(self, **kwargs: Any):
        return DummyPlaywrightContext()


class DummyPlaywright:
    def __init__(self):
        self.firefox = self
        self.chromium = self

    def launch(self, **kwargs: Any):
        return DummyPlaywrightBrowser()

    def launch_persistent_context(self, *args: Any, **kwargs: Any):
        return DummyPlaywrightContext()

    def start(self):
        return self

    def stop(self):
        return None


class DummySyncPlaywright:
    def start(self):
        return DummyPlaywright()


class DummyAudioDownloader:
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def download(self, url: str, output: Path) -> bool:
        output.write_bytes(b"audio")
        return True

"""Virtual-display lifecycle tests that do not require an X server."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from glyph_forge.live import virtual


class FakeDisplay:
    def __init__(self, **options) -> None:
        self.options = options
        self.new_display_var = ":42"
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def env(self) -> dict[str, str]:
        return {"DISPLAY": self.new_display_var, "SAFE": "1"}


def test_virtual_display_starts_and_stops_cleanly(monkeypatch) -> None:
    monkeypatch.setattr(virtual.platform, "system", lambda: "Linux")
    monkeypatch.setitem(
        sys.modules,
        "pyvirtualdisplay",
        SimpleNamespace(Display=FakeDisplay),
    )
    session = virtual.VirtualDisplaySession(800, 600)

    with session as active:
        display = active._display
        assert active.name == ":42"
        assert active.environment()["DISPLAY"] == ":42"
        assert display.options["size"] == (800, 600)

    assert display.stopped is True
    assert session.active is False


def test_virtual_display_rejects_unsupported_platform(monkeypatch) -> None:
    monkeypatch.setattr(virtual.platform, "system", lambda: "Windows")

    with pytest.raises(virtual.VirtualDisplayError, match="X11"):
        virtual.VirtualDisplaySession().start()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"height": 0},
        {"color_depth": 12},
        {"backend": "wayland"},
    ],
)
def test_virtual_display_validates_options(kwargs) -> None:
    with pytest.raises(ValueError):
        virtual.VirtualDisplaySession(**kwargs)

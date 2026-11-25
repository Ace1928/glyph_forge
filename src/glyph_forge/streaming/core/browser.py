"""
Browser Capture Module for Glyph Forge.

Captures video from browser-based streaming services (Netflix, etc.)
using Playwright for browser automation.

Requirements:
    pip install playwright
    playwright install firefox

Note: This is for personal/educational use only.
Respect content licensing and terms of service.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import numpy as np

try:
    from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


def get_firefox_profile_path() -> Optional[Path]:
    """Find the default Firefox profile path."""
    # Common Firefox profile locations
    possible_paths = [
        # Snap Firefox (Ubuntu)
        Path.home() / "snap" / "firefox" / "common" / ".mozilla" / "firefox",
        # Standard Linux
        Path.home() / ".mozilla" / "firefox",
        # macOS
        Path.home() / "Library" / "Application Support" / "Firefox" / "Profiles",
        # Windows
        Path.home() / "AppData" / "Roaming" / "Mozilla" / "Firefox" / "Profiles",
    ]

    for base_path in possible_paths:
        if base_path.exists():
            # Find default profile (usually ends with .default or .default-release)
            profiles = list(base_path.glob("*.default*"))
            if profiles:
                print(f"  Found Firefox profile: {profiles[0]}")
                return profiles[0]
            # Or just use the first profile directory found
            profiles = [
                p
                for p in base_path.iterdir()
                if p.is_dir()
                and not p.name.startswith(".")
                and not p.name.startswith("Crash")
                and not p.name.startswith("Pending")
            ]
            if profiles:
                print(f"  Found Firefox profile: {profiles[0]}")
                return profiles[0]

    return None


@dataclass
class BrowserConfig:
    """Browser capture configuration."""

    browser: str = "firefox"  # firefox, chromium
    headless: bool = False  # Netflix blocks headless
    width: int = 1920
    height: int = 1080
    capture_fps: float = 30.0
    user_data_dir: Optional[Path] = None  # For persistent login
    use_system_profile: bool = True  # Use existing Firefox profile


class BrowserCapture:
    """Captures video frames from browser streaming.

    Supports:
    - Netflix
    - YouTube (alternative to yt-dlp)
    - Disney+
    - Hulu
    - Any browser-based video

    Usage:
        cap = BrowserCapture('https://netflix.com/watch/...')
        cap.start()

        for frame in cap.frames():
            process(frame)

        cap.stop()
    """

    def __init__(self, url: str, config: Optional[BrowserConfig] = None):
        """Initialize browser capture.

        Args:
            url: URL to capture
            config: Browser configuration
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Browser capture requires playwright. "
                "Install with: pip install playwright && playwright install firefox"
            )

        self.url = url
        self.config = config or BrowserConfig()

        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._running = False
        self._frame_interval = 1.0 / self.config.capture_fps

    def start(self):
        """Start browser and navigate to URL."""
        self._playwright = sync_playwright().start()

        # Determine which browser and profile to use
        use_firefox = self.config.browser == "firefox"

        if use_firefox:
            print("🦊 Launching Firefox...")

            # Get Firefox profile path
            profile_path = None
            if self.config.use_system_profile:
                profile_path = self.config.user_data_dir or get_firefox_profile_path()
                if profile_path:
                    print(f"  Using profile: {profile_path}")

            if profile_path:
                # Use persistent context with existing Firefox profile
                self._context = self._playwright.firefox.launch_persistent_context(
                    str(profile_path),
                    headless=self.config.headless,
                    viewport={"width": self.config.width, "height": self.config.height},
                    slow_mo=50,  # Slow down for stability
                )
                self._page = (
                    self._context.pages[0] if self._context.pages else self._context.new_page()
                )
            else:
                # Launch fresh Firefox
                self._browser = self._playwright.firefox.launch(
                    headless=self.config.headless,
                )
                self._context = self._browser.new_context(
                    viewport={"width": self.config.width, "height": self.config.height}
                )
                self._page = self._context.new_page()
        else:
            # Chromium fallback
            print("🌐 Launching Chromium...")
            launch_args = [
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ]

            if self.config.user_data_dir:
                self._context = self._playwright.chromium.launch_persistent_context(
                    str(self.config.user_data_dir),
                    headless=self.config.headless,
                    args=launch_args,
                    viewport={"width": self.config.width, "height": self.config.height},
                )
                self._page = (
                    self._context.pages[0] if self._context.pages else self._context.new_page()
                )
            else:
                self._browser = self._playwright.chromium.launch(
                    headless=self.config.headless, args=launch_args
                )
                self._context = self._browser.new_context(
                    viewport={"width": self.config.width, "height": self.config.height}
                )
                self._page = self._context.new_page()

        # Navigate to URL
        print(f"🌐 Opening: {self.url[:60]}...")
        self._page.goto(self.url, timeout=60000)

        # Wait for video element to appear
        self._wait_for_video()

        self._running = True
        print("✓ Browser ready for capture")

    def _wait_for_video(self, timeout: float = 30.0):
        """Wait for video element to be ready."""
        try:
            # Try common video element selectors
            selectors = [
                "video",
                ".VideoContainer video",
                "#player video",
                ".video-player video",
            ]

            for selector in selectors:
                try:
                    self._page.wait_for_selector(
                        selector, timeout=int(timeout * 1000 / len(selectors))
                    )
                    print(f"✓ Found video element: {selector}")

                    # Click to start playback if needed
                    try:
                        play_button = self._page.query_selector('[aria-label*="Play"]')
                        if play_button:
                            play_button.click()
                    except Exception:
                        pass

                    return
                except Exception:
                    continue

            print("⚠ No video element found, capturing page anyway")

        except Exception as e:
            print(f"⚠ Video wait error: {e}")

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture current frame from browser.

        Returns:
            Frame as BGR numpy array, or None on error
        """
        if not self._running or not self._page:
            return None

        try:
            # Take screenshot
            screenshot_bytes = self._page.screenshot(type="png")

            # Convert to numpy array
            import cv2

            nparr = np.frombuffer(screenshot_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            return frame

        except Exception as e:
            print(f"Capture error: {e}")
            return None

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Generator yielding captured frames.

        Yields:
            BGR frames as numpy arrays
        """
        last_capture = 0

        while self._running:
            now = time.perf_counter()

            # Rate limiting
            if now - last_capture < self._frame_interval:
                time.sleep(self._frame_interval - (now - last_capture))

            frame = self.capture_frame()
            if frame is not None:
                yield frame

            last_capture = time.perf_counter()

    def stop(self):
        """Stop browser capture and cleanup."""
        self._running = False

        if self._context:
            try:
                self._context.close()
            except Exception:
                pass

        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass

        if self._playwright:
            try:
                self._playwright.stop()
            except Exception:
                pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class NetflixCapture(BrowserCapture):
    """Specialized capture for Netflix.

    Handles Netflix-specific UI interactions:
    - Skip intro button
    - Full screen mode
    - Hide controls

    Uses system Firefox profile by default for existing login.
    """

    def __init__(self, url: str, config: Optional[BrowserConfig] = None):
        """Initialize Netflix capture with Firefox defaults."""
        if config is None:
            config = BrowserConfig(
                browser="firefox",
                use_system_profile=True,
                headless=False,
            )
        super().__init__(url, config)

    def _wait_for_video(self, timeout: float = 30.0):
        """Wait for Netflix video player."""
        try:
            # Wait for Netflix player
            self._page.wait_for_selector(".watch-video", timeout=int(timeout * 1000))
            print("✓ Netflix player detected")

            # Hide UI elements for clean capture
            self._page.evaluate("""
                // Hide Netflix controls
                const style = document.createElement('style');
                style.textContent = `
                    .watch-video--bottom-controls-container,
                    .watch-video--evidence-overlay,
                    .ltr-omkt8s,
                    .watch-video--flag-audio-description {
                        opacity: 0 !important;
                    }
                `;
                document.head.appendChild(style);
            """)

            # Click play if paused
            try:
                play_button = self._page.query_selector('[data-uia="player-play-button"]')
                if play_button:
                    play_button.click()
            except Exception:
                pass

            # Skip intro if available
            time.sleep(3)
            try:
                skip_button = self._page.query_selector('[data-uia="player-skip-intro"]')
                if skip_button:
                    skip_button.click()
                    print("⏭ Skipped intro")
            except Exception:
                pass

        except Exception as e:
            print(f"⚠ Netflix detection error: {e}")


def create_browser_capture(url: str, config: Optional[BrowserConfig] = None) -> BrowserCapture:
    """Factory function to create appropriate browser capture.

    Args:
        url: URL to capture
        config: Browser configuration

    Returns:
        Appropriate BrowserCapture subclass
    """
    url_lower = url.lower()

    if "netflix.com" in url_lower:
        return NetflixCapture(url, config)
    else:
        return BrowserCapture(url, config)

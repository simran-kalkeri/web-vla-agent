"""
Playwright-based Browser Environment.

Provides a gym-like interface for the VLA agent to interact with
real web pages:  reset → step → observe → repeat.

Supports: click, type, scroll, select actions on DOM elements.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from utils.config import EnvironmentConfig


# ── Observation ──────────────────────────────────────────────

@dataclass
class Observation:
    """Structured observation returned after each environment step."""
    screenshot: Optional[Image.Image] = None
    dom_tree: str = ""
    url: str = ""
    page_title: str = ""
    error: Optional[str] = None
    done: bool = False


# ── Action definitions ───────────────────────────────────────

ACTION_TYPES = ["click", "type", "scroll", "select"]


@dataclass
class WebAction:
    """A single action to execute in the browser."""
    action_type: str = "click"       # click / type / scroll / select
    element_selector: str = ""       # CSS selector or XPath
    value: str = ""                  # text to type or option to select
    scroll_direction: str = "down"   # up / down (for scroll)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.action_type,
            "element": self.element_selector,
            "value": self.value,
            "scroll_direction": self.scroll_direction,
        }


# ── Browser Environment ─────────────────────────────────────

class BrowserEnvironment:
    """
    Playwright-backed browser environment with a step-based interface.

    The environment supports two modes:
      1. **Live mode** (default): Controls a real browser via Playwright.
      2. **Mock mode** (``use_mock=True``): Returns dummy observations for
         testing the pipeline without a browser.

    Parameters
    ----------
    config : EnvironmentConfig | None
        Environment-specific settings.
    use_mock : bool
        If True, skip Playwright entirely and return mock observations.
    """

    def __init__(
        self,
        config: Optional[EnvironmentConfig] = None,
        use_mock: bool = False,
    ):
        self.config = config or EnvironmentConfig()
        self.use_mock = use_mock

        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._step_count = 0

    # ── Lifecycle ────────────────────────────────────────────

    async def start(self) -> None:
        """Launch the browser (no-op in mock mode)."""
        if self.use_mock:
            return

        from playwright.async_api import async_playwright
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless
        )
        self._context = await self._browser.new_context(
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            }
        )
        self._page = await self._context.new_page()

    async def close(self) -> None:
        """Shut down the browser."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    # ── Core API ─────────────────────────────────────────────

    async def reset(self, url: str = "about:blank") -> Observation:
        """Navigate to *url* and return the initial observation."""
        self._step_count = 0
        if self.use_mock:
            return self._mock_observation(url)

        assert self._page is not None, "Call start() first."
        await self._page.goto(url, timeout=self.config.timeout_ms, wait_until="domcontentloaded")
        return await self._observe()

    async def step(self, action: WebAction) -> Observation:
        """Execute *action* and return the resulting observation."""
        self._step_count += 1
        if self.use_mock:
            return self._mock_observation(f"mock://step-{self._step_count}")

        assert self._page is not None, "Call start() first."

        try:
            await self._execute_action(action)
        except Exception as e:
            obs = await self._observe()
            obs.error = str(e)
            return obs

        # Wait for navigation / network idle
        try:
            await self._page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass

        return await self._observe()

    # ── Observation helpers ──────────────────────────────────

    async def _observe(self) -> Observation:
        assert self._page is not None
        screenshot_bytes = await self._page.screenshot(type="png")
        screenshot = Image.open(BytesIO(screenshot_bytes))

        dom_tree = await self._page.content()
        url = self._page.url
        title = await self._page.title()

        return Observation(
            screenshot=screenshot,
            dom_tree=dom_tree,
            url=url,
            page_title=title,
        )

    async def get_screenshot(self) -> Optional[Image.Image]:
        if self.use_mock:
            return Image.new("RGB", (self.config.screenshot_width, self.config.screenshot_height))
        if self._page:
            data = await self._page.screenshot(type="png")
            return Image.open(BytesIO(data))
        return None

    async def get_dom(self) -> str:
        if self.use_mock:
            return "<html><body>Mock</body></html>"
        if self._page:
            return await self._page.content()
        return ""

    async def get_url(self) -> str:
        if self.use_mock:
            return "mock://page"
        if self._page:
            return self._page.url
        return ""

    # ── Action execution ─────────────────────────────────────

    async def _execute_action(self, action: WebAction) -> None:
        assert self._page is not None

        if action.action_type == "click":
            await self._click(action.element_selector)

        elif action.action_type == "type":
            await self._type(action.element_selector, action.value)

        elif action.action_type == "scroll":
            await self._scroll(action.scroll_direction)

        elif action.action_type == "select":
            await self._select(action.element_selector, action.value)

        else:
            raise ValueError(f"Unknown action type: {action.action_type}")

    async def _click(self, selector: str) -> None:
        assert self._page is not None
        try:
            # Try CSS first, then XPath, then text
            await self._page.click(selector, timeout=5000)
        except Exception:
            # Fallback: try as text
            await self._page.click(f"text={selector}", timeout=5000)

    async def _type(self, selector: str, text: str) -> None:
        assert self._page is not None
        try:
            await self._page.fill(selector, text, timeout=5000)
        except Exception:
            # Fallback: click + keyboard type
            await self._page.click(selector, timeout=5000)
            await self._page.keyboard.type(text)

    async def _scroll(self, direction: str) -> None:
        assert self._page is not None
        delta = -300 if direction == "up" else 300
        await self._page.mouse.wheel(0, delta)

    async def _select(self, selector: str, value: str) -> None:
        assert self._page is not None
        await self._page.select_option(selector, label=value, timeout=5000)

    # ── Mock helpers ─────────────────────────────────────────

    def _mock_observation(self, url: str = "mock://page") -> Observation:
        # Simulate realistic page DOM that changes per step
        if self._step_count == 0:
            dom = '''<html><head><title>Travel Booking</title></head><body>
<nav><a href="/" id="home">Home</a> <a href="/flights" id="flights-link">Flights</a>
<a href="/hotels" id="hotels-link">Hotels</a></nav>
<div id="main"><h1>Book Your Trip</h1>
<form id="search-form"><label for="from">From</label>
<input type="text" id="from" name="from" placeholder="Departure city"/>
<label for="to">To</label>
<input type="text" id="to" name="to" placeholder="Arrival city"/>
<label for="date">Date</label>
<input type="date" id="date" name="date"/>
<button type="submit" id="search-btn">Search Flights</button></form>
<div id="results"></div></div>
<footer><a href="/help">Help</a></footer></body></html>'''
            title = "Travel Booking"
        elif self._step_count <= 3:
            dom = '''<html><head><title>Flight Results</title></head><body>
<nav><a href="/">Home</a> <a href="/flights">Flights</a></nav>
<div id="main"><h1>Flight Results</h1>
<div id="filter"><select id="sort" name="sort"><option>Price</option><option>Duration</option></select>
<button id="filter-btn">Apply</button></div>
<div class="flight-card" id="flight-1"><span>NYC → LA</span>
<span class="price">$199</span><button id="book-1">Book Now</button></div>
<div class="flight-card" id="flight-2"><span>NYC → LA</span>
<span class="price">$249</span><button id="book-2">Book Now</button></div>
<button id="next-page">Next Page</button></div></body></html>'''
            title = "Flight Results"
        else:
            dom = '''<html><head><title>Booking Confirmed</title></head><body>
<div id="main"><h1>Booking Confirmed</h1>
<p>Your flight from NYC to LA has been booked.</p>
<p>Confirmation #: ABC123</p>
<a href="/" id="home-link">Return Home</a></div></body></html>'''
            title = "Booking Confirmed"
        return Observation(
            screenshot=Image.new(
                "RGB",
                (self.config.screenshot_width, self.config.screenshot_height),
                color=(200, 200, 200),
            ),
            dom_tree=dom,
            url=url,
            page_title=title,
        )

    # ── Properties ───────────────────────────────────────────

    @property
    def step_count(self) -> int:
        return self._step_count

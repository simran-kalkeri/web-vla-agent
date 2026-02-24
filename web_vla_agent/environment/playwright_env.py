"""
Playwright Browser Environment — Full State Extraction.

Provides a gym-like interface for the VLA agent with:
  - Full state extraction: DOM tree, screenshot, viewport info, action history
  - Bounding box extraction via JS evaluation
  - Interactable element detection
  - CLICK, TYPE, SELECT, SCROLL action execution
  - Step loop with max-step cap, timeout, exception handling
"""
from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from environment.dom_serializer import DOMSerializer


# ── State representation ─────────────────────────────────────

@dataclass
class BrowserState:
    """Full browser state returned after each step."""
    dom_tree: str = ""                          # Raw HTML
    screenshot: Optional[Image.Image] = None    # Full-page screenshot
    viewport_info: Dict[str, Any] = field(default_factory=dict)  # bbox, scroll pos
    action_history: List[Dict[str, Any]] = field(default_factory=list)  # prev actions
    url: str = ""
    page_title: str = ""
    serialized_dom: str = ""                    # Structured DOM tokens
    dom_elements: List[Dict[str, Any]] = field(default_factory=list)  # Element list
    error: Optional[str] = None
    done: bool = False
    step_number: int = 0


# ── Action definitions ───────────────────────────────────────

ACTION_TYPES = ["CLICK", "TYPE", "SELECT", "SCROLL"]


@dataclass
class WebAction:
    """A single action to execute in the browser."""
    action: str = "CLICK"           # CLICK / TYPE / SCROLL / SELECT
    element_id: int = -1            # DOM node ID
    value: str = ""                 # text to type or option to select
    direction: str = "down"         # up / down (for SCROLL)
    amount: int = 300               # scroll amount in pixels

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"action": self.action, "element_id": self.element_id}
        if self.action == "TYPE":
            d["value"] = self.value
        elif self.action == "SELECT":
            d["value"] = self.value
        elif self.action == "SCROLL":
            d["direction"] = self.direction
            d["amount"] = self.amount
        return d

    def to_history_string(self) -> str:
        """Human-readable action string for action history."""
        if self.action == "CLICK":
            return f"CLICK element_id={self.element_id}"
        elif self.action == "TYPE":
            return f'TYPE element_id={self.element_id} value="{self.value}"'
        elif self.action == "SELECT":
            return f'SELECT element_id={self.element_id} value="{self.value}"'
        elif self.action == "SCROLL":
            return f"SCROLL direction={self.direction} amount={self.amount}"
        return f"{self.action} element_id={self.element_id}"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WebAction":
        """Parse an action dict (from model output) into a WebAction."""
        return cls(
            action=d.get("action", "CLICK").upper(),
            element_id=int(d.get("element_id", -1)),
            value=d.get("value", ""),
            direction=d.get("direction", "down"),
            amount=int(d.get("amount", 300)),
        )


# ── JS for element extraction ────────────────────────────────

_EXTRACT_ELEMENTS_JS = """
() => {
    const results = [];
    const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_ELEMENT,
        null,
        false
    );
    let nodeId = 0;
    const interactableTags = new Set([
        'a', 'button', 'input', 'select', 'textarea', 'option',
        'label', 'summary', 'details', 'area'
    ]);
    const interactableRoles = new Set([
        'button', 'link', 'menuitem', 'tab', 'checkbox', 'radio',
        'switch', 'option', 'combobox', 'listbox', 'searchbox',
        'slider', 'spinbutton', 'textbox', 'treeitem'
    ]);
    const skipTags = new Set([
        'script', 'style', 'noscript', 'svg', 'path', 'meta', 'link'
    ]);

    let node = walker.currentNode;
    while (node && nodeId < 600) {
        const tag = node.tagName ? node.tagName.toLowerCase() : '';
        if (tag && !skipTags.has(tag)) {
            const rect = node.getBoundingClientRect();
            const style = window.getComputedStyle(node);
            const isVisible = (
                style.display !== 'none' &&
                style.visibility !== 'hidden' &&
                style.opacity !== '0' &&
                rect.width > 0 && rect.height > 0
            );

            if (isVisible) {
                const role = (node.getAttribute('role') || '').toLowerCase();
                const isInteractable = (
                    interactableTags.has(tag) ||
                    interactableRoles.has(role) ||
                    node.hasAttribute('onclick') ||
                    node.hasAttribute('tabindex')
                );

                // Get direct text content (not from children)
                let text = '';
                for (const child of node.childNodes) {
                    if (child.nodeType === Node.TEXT_NODE) {
                        const t = child.textContent.trim();
                        if (t) text += (text ? ' ' : '') + t;
                    }
                }

                // Collect key attributes
                const attrs = {};
                for (const attr of ['id', 'class', 'role', 'type', 'placeholder',
                    'aria-label', 'href', 'name', 'value', 'title', 'alt',
                    'for', 'action', 'src']) {
                    const val = node.getAttribute(attr);
                    if (val) attrs[attr] = val.substring(0, 100);
                }

                // Compute depth
                let depth = 0;
                let parent = node.parentElement;
                while (parent) { depth++; parent = parent.parentElement; }

                results.push({
                    node_id: nodeId,
                    tag: tag,
                    text: text.substring(0, 200),
                    attributes: attrs,
                    bbox: [
                        rect.x + window.scrollX,
                        rect.y + window.scrollY,
                        rect.width,
                        rect.height
                    ],
                    depth: depth,
                    is_interactable: isInteractable,
                    is_visible: true
                });
            }
        }
        nodeId++;
        node = walker.nextNode();
    }
    return results;
}
"""

_GET_VIEWPORT_JS = """
() => {
    return {
        width: window.innerWidth,
        height: window.innerHeight,
        scrollX: window.scrollX,
        scrollY: window.scrollY,
        pageWidth: document.documentElement.scrollWidth,
        pageHeight: document.documentElement.scrollHeight
    };
}
"""


# ── Browser Environment ─────────────────────────────────────

class BrowserEnvironment:
    """
    Playwright-backed browser environment with full state extraction.

    Modes:
      - **Live mode**: Controls a real browser via Playwright
      - **Mock mode** (use_mock=True): Returns dummy states for testing

    Parameters
    ----------
    max_steps : int
        Maximum number of steps before forced termination.
    timeout : int
        Action timeout in milliseconds.
    headless : bool
        Run browser in headless mode.
    viewport_width : int
    viewport_height : int
    use_mock : bool
        If True, skip Playwright entirely.
    """

    def __init__(
        self,
        max_steps: int = 30,
        timeout: int = 30000,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        use_mock: bool = False,
    ):
        self.max_steps = max_steps
        self.timeout = timeout
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.use_mock = use_mock

        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._step_count = 0
        self._action_history: List[Dict[str, Any]] = []
        self._start_time: float = 0.0

        self._dom_serializer = DOMSerializer(
            max_nodes=500,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
        )

    # ── Lifecycle ────────────────────────────────────────────

    async def start(self) -> None:
        """Launch the browser."""
        if self.use_mock:
            return
        from playwright.async_api import async_playwright
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
        )
        self._context = await self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
        )
        self._page = await self._context.new_page()

    async def close(self) -> None:
        """Shut down the browser."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._browser = None
        self._playwright = None

    # ── Core API ─────────────────────────────────────────────

    async def reset(self, url: str = "about:blank") -> BrowserState:
        """Navigate to url and return the initial state."""
        self._step_count = 0
        self._action_history = []
        self._start_time = time.time()

        if self.use_mock:
            return self._mock_state(url)

        assert self._page is not None, "Call start() first."
        try:
            await self._page.goto(
                url, timeout=self.timeout, wait_until="domcontentloaded",
            )
        except Exception as e:
            state = BrowserState(error=str(e), url=url)
            return state

        return await self.extract_state()

    async def step(self, action: WebAction) -> BrowserState:
        """Execute action and return the resulting state."""
        self._step_count += 1

        # Check max steps
        if self._step_count > self.max_steps:
            state = await self.extract_state() if not self.use_mock else self._mock_state("")
            state.done = True
            state.error = f"Max steps ({self.max_steps}) exceeded"
            return state

        # Check timeout
        elapsed = time.time() - self._start_time if self._start_time else 0
        if elapsed > self.timeout / 1000 * 2:  # 2x action timeout as global timeout
            state = await self.extract_state() if not self.use_mock else self._mock_state("")
            state.done = True
            state.error = f"Global timeout ({elapsed:.0f}s)"
            return state

        # Record action in history
        self._action_history.append({
            "step": self._step_count,
            **action.to_dict(),
            "description": action.to_history_string(),
        })

        if self.use_mock:
            return self._mock_state(f"mock://step-{self._step_count}")

        assert self._page is not None, "Call start() first."

        # Execute action with exception handling
        try:
            await self._execute_action(action)
        except Exception as e:
            state = await self.extract_state()
            state.error = f"Action failed: {str(e)}"
            return state

        # Wait for page to settle
        try:
            await self._page.wait_for_load_state(
                "domcontentloaded", timeout=5000,
            )
        except Exception:
            pass

        return await self.extract_state()

    async def extract_state(self) -> BrowserState:
        """
        Extract full browser state.

        Returns state with:
          - dom_tree: raw HTML
          - screenshot: PIL Image
          - viewport_info: dimensions + scroll position
          - action_history: list of previous actions
          - serialized_dom: structured DOM tokens
          - dom_elements: raw element list
        """
        if self.use_mock:
            return self._mock_state("")

        assert self._page is not None

        # Extract screenshot
        screenshot = None
        try:
            screenshot_bytes = await self._page.screenshot(type="png")
            screenshot = Image.open(BytesIO(screenshot_bytes))
        except Exception:
            screenshot = Image.new("RGB", (self.viewport_width, self.viewport_height))

        # Extract raw DOM
        try:
            dom_tree = await self._page.content()
        except Exception:
            dom_tree = ""

        # Extract viewport info
        try:
            viewport_info = await self._page.evaluate(_GET_VIEWPORT_JS)
        except Exception:
            viewport_info = {
                "width": self.viewport_width,
                "height": self.viewport_height,
                "scrollX": 0, "scrollY": 0,
            }

        # Extract DOM elements with bounding boxes
        try:
            dom_elements = await self._page.evaluate(_EXTRACT_ELEMENTS_JS)
        except Exception:
            dom_elements = []

        # Serialize DOM into structured tokens
        serialized_dom, _ = self._dom_serializer.serialize_dom(dom_elements)

        # URL and title
        try:
            url = self._page.url
            title = await self._page.title()
        except Exception:
            url = ""
            title = ""

        return BrowserState(
            dom_tree=dom_tree,
            screenshot=screenshot,
            viewport_info=viewport_info,
            action_history=list(self._action_history),
            url=url,
            page_title=title,
            serialized_dom=serialized_dom,
            dom_elements=dom_elements,
            step_number=self._step_count,
        )

    # ── Action execution ─────────────────────────────────────

    async def _execute_action(self, action: WebAction) -> None:
        """Execute a single action in the browser."""
        assert self._page is not None

        if action.action == "CLICK":
            await self._do_click(action.element_id)
        elif action.action == "TYPE":
            await self._do_type(action.element_id, action.value)
        elif action.action == "SELECT":
            await self._do_select(action.element_id, action.value)
        elif action.action == "SCROLL":
            await self._do_scroll(action.direction, action.amount)
        else:
            raise ValueError(f"Unknown action type: {action.action}")

    async def _do_click(self, element_id: int) -> None:
        """Click on element by node_id."""
        assert self._page is not None
        # Use JS to find and click element by our assigned node_id
        js = f"""
        () => {{
            const walker = document.createTreeWalker(
                document.body, NodeFilter.SHOW_ELEMENT, null, false
            );
            let nodeId = 0;
            let node = walker.currentNode;
            while (node && nodeId <= {element_id}) {{
                if (nodeId === {element_id}) {{
                    node.click();
                    return true;
                }}
                nodeId++;
                node = walker.nextNode();
            }}
            return false;
        }}
        """
        result = await self._page.evaluate(js)
        if not result:
            raise ValueError(f"Element with id={element_id} not found")

    async def _do_type(self, element_id: int, value: str) -> None:
        """Type text into element by node_id."""
        assert self._page is not None
        escaped_value = value.replace("'", "\\'").replace('"', '\\"')
        js = f"""
        () => {{
            const walker = document.createTreeWalker(
                document.body, NodeFilter.SHOW_ELEMENT, null, false
            );
            let nodeId = 0;
            let node = walker.currentNode;
            while (node && nodeId <= {element_id}) {{
                if (nodeId === {element_id}) {{
                    node.focus();
                    node.value = "{escaped_value}";
                    node.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    node.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    return true;
                }}
                nodeId++;
                node = walker.nextNode();
            }}
            return false;
        }}
        """
        result = await self._page.evaluate(js)
        if not result:
            raise ValueError(f"Element with id={element_id} not found for typing")

    async def _do_select(self, element_id: int, value: str) -> None:
        """Select option in element by node_id."""
        assert self._page is not None
        escaped_value = value.replace("'", "\\'").replace('"', '\\"')
        js = f"""
        () => {{
            const walker = document.createTreeWalker(
                document.body, NodeFilter.SHOW_ELEMENT, null, false
            );
            let nodeId = 0;
            let node = walker.currentNode;
            while (node && nodeId <= {element_id}) {{
                if (nodeId === {element_id}) {{
                    for (const opt of node.options || []) {{
                        if (opt.text === "{escaped_value}" || opt.value === "{escaped_value}") {{
                            opt.selected = true;
                            node.dispatchEvent(new Event('change', {{ bubbles: true }}));
                            return true;
                        }}
                    }}
                    return false;
                }}
                nodeId++;
                node = walker.nextNode();
            }}
            return false;
        }}
        """
        await self._page.evaluate(js)

    async def _do_scroll(self, direction: str, amount: int) -> None:
        """Scroll the page."""
        assert self._page is not None
        delta = -amount if direction == "up" else amount
        await self._page.mouse.wheel(0, delta)

    # ── Mock mode ────────────────────────────────────────────

    def _mock_state(self, url: str = "mock://page") -> BrowserState:
        """Generate a mock state for testing without a browser."""
        if self._step_count == 0:
            html = '''<html><head><title>Travel Booking</title></head><body>
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
            html = '''<html><head><title>Flight Results</title></head><body>
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
            html = '''<html><head><title>Booking Confirmed</title></head><body>
<div id="main"><h1>Booking Confirmed</h1>
<p>Your flight from NYC to LA has been booked.</p>
<p>Confirmation #: ABC123</p>
<a href="/" id="home-link">Return Home</a></div></body></html>'''
            title = "Booking Confirmed"

        # Serialize DOM from mock HTML
        serialized_dom, nodes = self._dom_serializer.serialize_from_html(html)

        # Build mock elements list
        mock_elements = [
            {
                "node_id": n.node_id, "tag": n.tag, "text": n.text,
                "attributes": n.attributes, "bbox": list(n.bbox) if n.bbox else None,
                "depth": n.depth, "is_interactable": n.is_interactable,
            }
            for n in nodes
        ]

        return BrowserState(
            dom_tree=html,
            screenshot=Image.new("RGB", (self.viewport_width, self.viewport_height), (200, 200, 200)),
            viewport_info={
                "width": self.viewport_width, "height": self.viewport_height,
                "scrollX": 0, "scrollY": 0,
            },
            action_history=list(self._action_history),
            url=url or "mock://page",
            page_title=title,
            serialized_dom=serialized_dom,
            dom_elements=mock_elements,
            step_number=self._step_count,
        )

    # ── Properties ───────────────────────────────────────────

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def action_history(self) -> List[Dict[str, Any]]:
        return list(self._action_history)

    def get_action_history_text(self) -> str:
        """Get action history as formatted text for model input."""
        if not self._action_history:
            return "No previous actions."
        lines = []
        for entry in self._action_history:
            lines.append(f"Step {entry['step']}: {entry['description']}")
        return "\n".join(lines)

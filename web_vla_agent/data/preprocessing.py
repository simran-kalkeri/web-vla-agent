"""
Data preprocessing for the VLA Web Agent.

- DOMProcessor: parse raw HTML → list of DOMElement dataclass objects
- ScreenshotProcessor: resize / normalise screenshots
- crop_element_from_screenshot: visual patch extraction per element
- build_dom_graph: parent–child adjacency from parsed DOM
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from bs4 import BeautifulSoup, Tag
from PIL import Image


# ── Data structures ──────────────────────────────────────────

@dataclass
class BoundingBox:
    """Axis-aligned bounding box (pixel coords, origin top-left)."""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class DOMElement:
    """Structured representation of a single DOM node."""
    element_id: str = ""
    tag: str = ""
    text: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    bounding_box: Optional[BoundingBox] = None
    is_clickable: bool = False
    is_visible: bool = True
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0

    @property
    def signature(self) -> str:
        """Short text signature for embedding: 'tag | text | key-attrs'."""
        attr_str = " ".join(
            f'{k}="{v}"'
            for k, v in sorted(self.attributes.items())
            if k in {"id", "class", "name", "type", "placeholder", "aria-label", "role", "href", "value"}
        )
        text_short = (self.text[:80] + "…") if len(self.text) > 80 else self.text
        return f"{self.tag} | {text_short} | {attr_str}".strip(" |")


@dataclass
class DOMGraph:
    """Adjacency-list representation of the DOM tree."""
    nodes: List[DOMElement]
    adjacency: Dict[str, List[str]]  # parent → children


# ── Clickable heuristic ─────────────────────────────────────

_CLICKABLE_TAGS = frozenset({
    "a", "button", "input", "select", "textarea", "option",
    "label", "summary", "details",
})

_CLICKABLE_ROLES = frozenset({
    "button", "link", "menuitem", "tab", "checkbox", "radio",
    "switch", "option", "combobox", "listbox", "searchbox",
})


def _is_probably_clickable(tag: Tag) -> bool:
    if tag.name in _CLICKABLE_TAGS:
        return True
    role = (tag.get("role") or "").lower()
    if role in _CLICKABLE_ROLES:
        return True
    onclick = tag.get("onclick") or tag.get("ng-click") or tag.get("@click")
    if onclick:
        return True
    return False


# ── DOM processor ────────────────────────────────────────────

class DOMProcessor:
    """Parse raw HTML into a flat list of :class:`DOMElement` instances."""

    def __init__(self, max_elements: int = 128):
        self.max_elements = max_elements

    # ── public ───────────────────────────────────────────────
    def parse(self, html: str) -> List[DOMElement]:
        """Parse *html* and return structured DOM elements."""
        soup = BeautifulSoup(html, "lxml")
        elements: List[DOMElement] = []
        self._walk(soup, elements, parent_id=None, depth=0)
        return elements[: self.max_elements]

    def parse_with_graph(self, html: str) -> DOMGraph:
        """Parse and also return a graph with adjacency."""
        elements = self.parse(html)
        adjacency: Dict[str, List[str]] = {}
        for el in elements:
            adjacency.setdefault(el.element_id, [])
            if el.parent_id:
                adjacency.setdefault(el.parent_id, []).append(el.element_id)
        return DOMGraph(nodes=elements, adjacency=adjacency)

    # ── private ──────────────────────────────────────────────
    def _walk(
        self,
        node: Any,
        out: List[DOMElement],
        parent_id: Optional[str],
        depth: int,
    ) -> None:
        if len(out) >= self.max_elements:
            return
        if not isinstance(node, Tag):
            return
        # skip non-visual tags
        if node.name in {"script", "style", "noscript", "svg", "path", "meta", "link", "head"}:
            return

        eid = self._element_id(node, len(out))
        text = self._extract_text(node)
        attrs = {k: (v if isinstance(v, str) else " ".join(v))
                 for k, v in (node.attrs or {}).items()}

        # bounding-box placeholder (filled later if available from dataset)
        bbox = self._parse_bbox(attrs)

        elem = DOMElement(
            element_id=eid,
            tag=node.name,
            text=text,
            attributes=attrs,
            bounding_box=bbox,
            is_clickable=_is_probably_clickable(node),
            is_visible=self._is_visible(attrs),
            parent_id=parent_id,
            depth=depth,
        )
        out.append(elem)

        for child in node.children:
            self._walk(child, out, parent_id=eid, depth=depth + 1)

        # back-fill children ids
        elem.children_ids = [
            e.element_id for e in out if e.parent_id == eid
        ]

    @staticmethod
    def _element_id(tag: Tag, idx: int) -> str:
        raw = tag.get("data-backend-node-id") or tag.get("backend_node_id") or tag.get("id", "")
        if raw:
            return str(raw)
        # deterministic fallback
        sig = f"{tag.name}_{idx}"
        return hashlib.md5(sig.encode()).hexdigest()[:10]

    @staticmethod
    def _extract_text(tag: Tag) -> str:
        """Get direct text (not from children), stripped."""
        parts = []
        for child in tag.children:
            if isinstance(child, str):
                cleaned = child.strip()
                if cleaned:
                    parts.append(cleaned)
        return " ".join(parts)[:256]

    @staticmethod
    def _parse_bbox(attrs: Dict[str, str]) -> Optional[BoundingBox]:
        """Try to build a BoundingBox from data or style attributes."""
        # Mind2Web stores bbox as separate attributes in some variants
        try:
            if all(k in attrs for k in ("data-x", "data-y", "data-w", "data-h")):
                return BoundingBox(
                    x=float(attrs["data-x"]),
                    y=float(attrs["data-y"]),
                    width=float(attrs["data-w"]),
                    height=float(attrs["data-h"]),
                )
        except (ValueError, TypeError):
            pass
        return None

    @staticmethod
    def _is_visible(attrs: Dict[str, str]) -> bool:
        style = attrs.get("style", "")
        if "display: none" in style or "display:none" in style:
            return False
        if "visibility: hidden" in style or "visibility:hidden" in style:
            return False
        aria_hidden = attrs.get("aria-hidden", "").lower()
        if aria_hidden == "true":
            return False
        return True


# ── Screenshot processing ───────────────────────────────────

class ScreenshotProcessor:
    """Resize and normalise screenshots for vision encoders."""

    def __init__(self, size: Tuple[int, int] = (224, 224)):
        self.size = size

    def process(self, image: Image.Image) -> Image.Image:
        """Resize to target size, convert to RGB."""
        return image.convert("RGB").resize(self.size, Image.LANCZOS)

    def to_numpy(self, image: Image.Image) -> np.ndarray:
        """Process and return float32 array [C, H, W] in [0, 1]."""
        img = self.process(image)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)  # HWC → CHW


# ── Element cropping ────────────────────────────────────────

def crop_element_from_screenshot(
    screenshot: Image.Image,
    bbox: BoundingBox,
    target_size: Tuple[int, int] = (64, 64),
    padding: int = 4,
) -> Image.Image:
    """Crop the region around *bbox* from *screenshot* and resize."""
    W, H = screenshot.size
    x1, y1, x2, y2 = bbox.as_tuple()
    # clamp + padding
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(W, int(x2) + padding)
    y2 = min(H, int(y2) + padding)
    if x2 <= x1 or y2 <= y1:
        # degenerate box → return blank
        return Image.new("RGB", target_size, (128, 128, 128))
    crop = screenshot.crop((x1, y1, x2, y2))
    return crop.resize(target_size, Image.LANCZOS)


# ── DOM graph builder ────────────────────────────────────────

def build_dom_graph(elements: List[DOMElement]) -> DOMGraph:
    """Build a DOMGraph from an already-parsed element list."""
    adjacency: Dict[str, List[str]] = {}
    for el in elements:
        adjacency.setdefault(el.element_id, [])
        if el.parent_id and el.parent_id in adjacency:
            adjacency[el.parent_id].append(el.element_id)
    return DOMGraph(nodes=elements, adjacency=adjacency)

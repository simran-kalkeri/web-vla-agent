"""
Structured DOM Serialization for VLA Web Agent.

Serializes DOM nodes into structured token format for the multimodal
transformer. NOT flat text — structured per-node tokens with metadata.

Format:
  <node id=32 tag=button depth=3 bbox=(x,y,w,h)>
    text="Search and reserve a car"
    attributes="role=tab"
  </node>

Properties per node:
  - Tag name
  - Tree depth
  - Bounding box (x, y, width, height)
  - Visible text (truncated)
  - Key attributes (role, type, placeholder, aria-label, href, name, value)
  - Interactable flag
  - Unique element ID

Limits:
  - Top 500 visible nodes maximum
  - Sorted by viewport proximity (visible in viewport first)
  - Deterministic truncation — no random filtering
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SerializedNode:
    """A single serialized DOM node with all metadata."""
    node_id: int
    tag: str
    depth: int
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x, y, w, h) raw pixels
    bbox_norm: Optional[Tuple[float, float, float, float]] = None  # (x, y, w, h) normalized [0,1]
    text: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    is_interactable: bool = False
    is_visible: bool = True
    is_in_viewport: bool = False   # fully or partially in viewport
    viewport_distance: float = 0.0  # distance from viewport center


# Tags that are inherently interactable
_INTERACTABLE_TAGS = frozenset({
    "a", "button", "input", "select", "textarea", "option",
    "label", "summary", "details", "area",
})

# Roles that indicate interactability
_INTERACTABLE_ROLES = frozenset({
    "button", "link", "menuitem", "tab", "checkbox", "radio",
    "switch", "option", "combobox", "listbox", "searchbox",
    "slider", "spinbutton", "textbox", "treeitem",
})

# Attributes to preserve in serialization
_KEEP_ATTRIBUTES = frozenset({
    "role", "type", "placeholder", "aria-label", "aria_label",
    "href", "name", "value", "title", "alt", "id", "class",
    "for", "action", "method", "target", "src",
})

# Tags to skip entirely (non-visual)
_SKIP_TAGS = frozenset({
    "script", "style", "noscript", "svg", "path", "meta",
    "link", "head", "br", "hr",
})


class DOMSerializer:
    """
    Serialize DOM tree into structured node tokens for model input.

    Parameters
    ----------
    max_nodes : int
        Maximum number of nodes to include (default 500).
    max_text_len : int
        Maximum characters of visible text per node.
    max_attr_len : int
        Maximum characters per attribute value.
    viewport_width : int
        Viewport width for proximity sorting.
    viewport_height : int
        Viewport height for proximity sorting.
    """

    def __init__(
        self,
        max_nodes: int = 500,
        max_text_len: int = 200,
        max_attr_len: int = 100,
        viewport_width: int = 1280,
        viewport_height: int = 720,
    ):
        self.max_nodes = max_nodes
        self.max_text_len = max_text_len
        self.max_attr_len = max_attr_len
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

    def serialize_dom(
        self,
        elements: List[Dict[str, Any]],
    ) -> Tuple[str, List[SerializedNode]]:
        """
        Serialize a list of extracted DOM elements into structured text.

        Parameters
        ----------
        elements : list of dicts
            Each dict has: tag, text, attributes, bbox, depth, is_interactable, node_id

        Returns
        -------
        serialized_text : str
            Full serialized DOM for model input.
        nodes : list of SerializedNode
            Structured node objects for reference.
        """
        # Build SerializedNode objects
        nodes = []
        for i, el in enumerate(elements):
            tag = el.get("tag", "div").lower()
            if tag in _SKIP_TAGS:
                continue

            attrs = el.get("attributes", {})
            if isinstance(attrs, str):
                attrs = {}

            bbox = el.get("bbox")
            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                bbox = tuple(float(v) for v in bbox)
            else:
                bbox = None

            # Determine interactability
            is_interactable = (
                tag in _INTERACTABLE_TAGS
                or str(attrs.get("role", "")).lower() in _INTERACTABLE_ROLES
                or bool(attrs.get("onclick"))
                or bool(attrs.get("ng-click"))
                or bool(attrs.get("@click"))
                or el.get("is_interactable", False)
            )

            text = str(el.get("text", ""))[:self.max_text_len].strip()
            is_visible = el.get("is_visible", True)

            if not is_visible:
                continue

            # Compute viewport distance and in-viewport flag
            vp_dist = self._viewport_distance(bbox)
            in_viewport = self._is_in_viewport(bbox)

            # Normalize bbox to [0,1] relative to viewport
            bbox_norm = self._normalize_bbox(bbox)

            # Filter attributes to keep
            filtered_attrs = {}
            for k, v in attrs.items():
                if k in _KEEP_ATTRIBUTES and v:
                    filtered_attrs[k] = str(v)[:self.max_attr_len]

            node = SerializedNode(
                node_id=el.get("node_id", i),
                tag=tag,
                depth=el.get("depth", 0),
                bbox=bbox,
                bbox_norm=bbox_norm,
                text=text,
                attributes=filtered_attrs,
                is_interactable=is_interactable,
                is_visible=True,
                is_in_viewport=in_viewport,
                viewport_distance=vp_dist,
            )
            nodes.append(node)

        # Sort: in-viewport first, then interactable, then by viewport proximity
        # This ensures visible, actionable nodes are always prioritized
        nodes.sort(key=lambda n: (
            0 if n.is_in_viewport else 1,       # in-viewport first
            0 if n.is_interactable else 1,       # interactable second
            n.viewport_distance,                  # closest to center third
        ))

        # Truncate to max_nodes
        nodes = nodes[:self.max_nodes]

        # Build serialized text
        serialized_text = self._build_text(nodes)
        return serialized_text, nodes

    def serialize_from_html(
        self,
        html: str,
        element_bboxes: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
    ) -> Tuple[str, List[SerializedNode]]:
        """
        Parse raw HTML and serialize into structured tokens.

        Parameters
        ----------
        html : str
            Raw HTML string.
        element_bboxes : dict, optional
            Mapping from element index to (x, y, w, h) bounding boxes.

        Returns
        -------
        serialized_text : str
        nodes : list of SerializedNode
        """
        from bs4 import BeautifulSoup, Tag

        soup = BeautifulSoup(html, "lxml")
        elements = []
        self._walk_html(soup, elements, depth=0, counter=[0])

        # Apply bounding boxes if provided
        if element_bboxes:
            for el in elements:
                nid = el.get("node_id")
                if nid in element_bboxes:
                    el["bbox"] = element_bboxes[nid]

        return self.serialize_dom(elements)

    def _walk_html(
        self,
        node: Any,
        out: List[Dict[str, Any]],
        depth: int,
        counter: List[int],
    ) -> None:
        """Recursively walk HTML tree and extract elements."""
        from bs4 import Tag

        if not isinstance(node, Tag):
            return
        if node.name in _SKIP_TAGS:
            return

        node_id = counter[0]
        counter[0] += 1

        # Extract direct text (not from children)
        text_parts = []
        for child in node.children:
            if isinstance(child, str):
                cleaned = child.strip()
                if cleaned:
                    text_parts.append(cleaned)
        text = " ".join(text_parts)[:self.max_text_len]

        # Extract attributes
        attrs = {}
        for k, v in (node.attrs or {}).items():
            if isinstance(v, list):
                v = " ".join(v)
            attrs[k] = str(v)

        # Check visibility
        style = attrs.get("style", "")
        is_visible = not (
            "display:none" in style.replace(" ", "")
            or "visibility:hidden" in style.replace(" ", "")
            or attrs.get("aria-hidden", "").lower() == "true"
        )

        # Extract bbox from data attributes if available
        bbox = None
        try:
            bbox_rect = attrs.get("bounding_box_rect", "")
            if bbox_rect:
                parts = [float(x) for x in str(bbox_rect).split(",")[:4]]
                if len(parts) == 4:
                    bbox = tuple(parts)
        except (ValueError, TypeError):
            pass

        el = {
            "node_id": node_id,
            "tag": node.name,
            "text": text,
            "attributes": attrs,
            "bbox": bbox,
            "depth": depth,
            "is_visible": is_visible,
            "is_interactable": False,  # determined by serialize_dom
        }
        out.append(el)

        for child in node.children:
            self._walk_html(child, out, depth + 1, counter)

    def _viewport_distance(
        self,
        bbox: Optional[Tuple[float, float, float, float]],
    ) -> float:
        """Compute distance from viewport center. No bbox → max distance."""
        if bbox is None:
            return float("inf")

        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        vp_cx = self.viewport_width / 2
        vp_cy = self.viewport_height / 2

        return ((cx - vp_cx) ** 2 + (cy - vp_cy) ** 2) ** 0.5

    def _is_in_viewport(
        self,
        bbox: Optional[Tuple[float, float, float, float]],
    ) -> bool:
        """Check if element is at least partially within the viewport."""
        if bbox is None:
            return False
        x, y, w, h = bbox
        # Element is in viewport if it overlaps with (0, 0, vw, vh)
        return (
            x + w > 0 and x < self.viewport_width and
            y + h > 0 and y < self.viewport_height
        )

    def _normalize_bbox(
        self,
        bbox: Optional[Tuple[float, float, float, float]],
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Normalize bbox to [0, 1] relative to viewport dimensions.

        Clamps values so elements partially outside viewport stay in [0, 1].
        Uses 3 decimal places for consistent tokenization.
        """
        if bbox is None:
            return None
        x, y, w, h = bbox
        nx = round(max(0.0, min(1.0, x / self.viewport_width)), 3)
        ny = round(max(0.0, min(1.0, y / self.viewport_height)), 3)
        nw = round(max(0.0, min(1.0, w / self.viewport_width)), 3)
        nh = round(max(0.0, min(1.0, h / self.viewport_height)), 3)
        return (nx, ny, nw, nh)

    def _build_text(self, nodes: List[SerializedNode]) -> str:
        """
        Build the serialized text representation of all nodes.

        Uses NORMALIZED bbox values [0,1] for meaningful geometric signal.
        Format: bbox=(0.125,0.278,0.234,0.056)
        """
        lines = []
        for node in nodes:
            # Build node header with NORMALIZED bbox
            bbox_str = ""
            if node.bbox_norm:
                bbox_str = (
                    f" bbox=({node.bbox_norm[0]:.3f},"
                    f"{node.bbox_norm[1]:.3f},"
                    f"{node.bbox_norm[2]:.3f},"
                    f"{node.bbox_norm[3]:.3f})"
                )

            interactable_str = " interactable=true" if node.is_interactable else ""

            header = (
                f"<node id={node.node_id} tag={node.tag} "
                f"depth={node.depth}{bbox_str}{interactable_str}>"
            )

            # Build node body
            body_parts = []
            if node.text:
                escaped_text = node.text.replace('"', '\\"')
                body_parts.append(f'  text="{escaped_text}"')

            if node.attributes:
                attr_pairs = []
                for k, v in sorted(node.attributes.items()):
                    if k in _KEEP_ATTRIBUTES:
                        escaped_v = str(v).replace('"', '\\"')[:80]
                        attr_pairs.append(f'{k}="{escaped_v}"')
                if attr_pairs:
                    body_parts.append(f'  attributes="{" ".join(attr_pairs)}"')

            footer = "</node>"

            if body_parts:
                lines.append(header)
                lines.extend(body_parts)
                lines.append(footer)
            else:
                lines.append(f"{header}{footer}")

        return "\n".join(lines)

    @staticmethod
    def get_node_ids(nodes: List[SerializedNode]) -> List[int]:
        """Get all valid node IDs for action validation."""
        return [n.node_id for n in nodes]

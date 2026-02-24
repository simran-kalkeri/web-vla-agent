"""
Graph-Aware DOM Encoder — 1-Layer GCN (Refactored).

Replaces the original 2-layer GAT with a single GCNConv layer for
CPU-efficient, stable training while preserving DOM tree structure.

**Mathematical formulation**:
    Given DOM graph G = (V, E) with node features X ∈ R^{N×d}:

    h_i = σ(Σ_{j∈N(i)} (1/√(d_i·d_j)) W h_j)

    where d_i = deg(i) + 1 (self-loop augmented degree).

Tech: PyTorch Geometric (GCNConv)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

from data.preprocessing import DOMElement


# ── Edge types ───────────────────────────────────────────────

class EdgeType(IntEnum):
    PARENT_TO_CHILD = 0
    CHILD_TO_PARENT = 1
    SIBLING = 2


# ── Structural feature builder ──────────────────────────────

# Common HTML tags for one-hot encoding
_TAG_VOCAB = [
    "div", "span", "a", "button", "input", "select", "textarea",
    "form", "label", "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "table", "tr", "td", "th", "img", "nav",
    "header", "footer", "section", "article", "main", "other",
]
_TAG_TO_IDX = {t: i for i, t in enumerate(_TAG_VOCAB)}
_NUM_TAGS = len(_TAG_VOCAB)


def structural_features(elements: List[DOMElement]) -> torch.Tensor:
    """
    Build structural feature vector per DOM node.

    Features (per node):
      - tag one-hot       [31]
      - depth (normalised) [1]
      - is_clickable       [1]
      - is_visible         [1]
      - has_text           [1]
      - num_children       [1]
    Total: 36 dims
    """
    feats = []
    max_depth = max((e.depth for e in elements), default=1) or 1

    for el in elements:
        # Tag one-hot
        tag_idx = _TAG_TO_IDX.get(el.tag, _TAG_TO_IDX["other"])
        tag_oh = [0.0] * _NUM_TAGS
        tag_oh[tag_idx] = 1.0

        feats.append(
            tag_oh
            + [
                el.depth / max_depth,
                float(el.is_clickable),
                float(el.is_visible),
                float(len(el.text) > 0),
                min(len(el.children_ids) / 10.0, 1.0),
            ]
        )

    return torch.tensor(feats, dtype=torch.float32)


STRUCTURAL_DIM = _NUM_TAGS + 5  # 36


# ── Graph construction ──────────────────────────────────────

def build_edge_index(
    elements: List[DOMElement],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge_index ``[2, E]`` and edge_type ``[E]`` from DOM elements.

    Edge types:
      0 = parent → child
      1 = child → parent
      2 = sibling (same parent, undirected)

    Returns
    -------
    edge_index : [2, E] long tensor
    edge_type  : [E]    long tensor
    """
    id_to_idx: Dict[str, int] = {el.element_id: i for i, el in enumerate(elements)}
    src, dst, etypes = [], [], []

    for i, el in enumerate(elements):
        for cid in el.children_ids:
            if cid in id_to_idx:
                j = id_to_idx[cid]
                src.append(i); dst.append(j); etypes.append(EdgeType.PARENT_TO_CHILD)
                src.append(j); dst.append(i); etypes.append(EdgeType.CHILD_TO_PARENT)

    # Sibling edges
    parent_groups: Dict[Optional[str], List[int]] = {}
    for i, el in enumerate(elements):
        parent_groups.setdefault(el.parent_id, []).append(i)

    for siblings in parent_groups.values():
        if len(siblings) < 2:
            continue
        for a_idx in range(len(siblings)):
            for b_idx in range(a_idx + 1, len(siblings)):
                i, j = siblings[a_idx], siblings[b_idx]
                src.append(i); dst.append(j); etypes.append(EdgeType.SIBLING)
                src.append(j); dst.append(i); etypes.append(EdgeType.SIBLING)

    if not src:
        edge_index = torch.zeros(2, 1, dtype=torch.long)
        edge_type = torch.zeros(1, dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor(etypes, dtype=torch.long)

    return edge_index, edge_type


# ── Graph DOM Encoder (1-Layer GCN) ──────────────────────────

class GraphDOMEncoder(nn.Module):
    """
    Graph-Aware DOM Encoder using 1-layer GCNConv.

    Combines text embeddings with structural features, then applies
    a single GCN layer to produce context-aware node embeddings.

    **Architecture**:
        Input: text_emb [N, text_dim] + struct_feat [N, struct_dim]
        → Linear projection → [N, hidden_dim]
        → GCNConv (with self-loops) → [N, hidden_dim]
        → Residual + LayerNorm → output [N, hidden_dim]

    Parameters
    ----------
    text_dim : int
        Dimension of input text embeddings (384 for MiniLM).
    struct_dim : int
        Dimension of structural features (default 36).
    hidden_dim : int
        Dimension of output node embeddings (256).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        text_dim: int = 384,
        struct_dim: int = STRUCTURAL_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project text + structural features to hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(text_dim + struct_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Single GCN layer
        self.gcn = GCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            add_self_loops=True,
            normalize=True,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_emb: torch.Tensor,     # [N, text_dim]
        struct_feat: torch.Tensor,   # [N, struct_dim]
        edge_index: torch.Tensor,    # [2, E]
    ) -> torch.Tensor:
        """
        Produce graph-contextualized DOM node embeddings.

        Returns
        -------
        [N, hidden_dim]
        """
        # Concatenate text + structural features
        x = torch.cat([text_emb, struct_feat], dim=-1)
        x = self.input_proj(x)  # [N, hidden_dim]

        # Single GCN layer with residual connection
        x_res = x
        x = self.gcn(x, edge_index)
        x = F.gelu(x)
        x = self.dropout(x)
        x = x + x_res  # Residual

        x = self.layer_norm(x)
        return x  # [N, hidden_dim]

    def encode_dom(
        self,
        elements: List[DOMElement],
        text_emb: torch.Tensor,  # [N, text_dim]
    ) -> torch.Tensor:
        """
        Convenience: build graph from elements and encode.

        Parameters
        ----------
        elements : list of DOMElement
        text_emb : [N, text_dim] pre-computed text embeddings

        Returns
        -------
        [N, hidden_dim]
        """
        struct_feat = structural_features(elements).to(text_emb.device)
        edge_index, _ = build_edge_index(elements)
        edge_index = edge_index.to(text_emb.device)
        return self.forward(text_emb, struct_feat, edge_index)

"""
Unified VLA Grounder — Single Cross-Attention Block (Refactored).

Replaces the original 2-layer cross-attention grounder + separate
ActionPolicyNetwork with a single unified module.

**Architecture**:
    Subgoal embedding g + GCN node embeddings {h_i} + visual embeddings {v_i}
    → 1-layer Cross-Attention Transformer (256 dim, 4 heads)
    → Element logits: softmax(W·z_i)
    → Action logits:  softmax(W_a·z_top)
    → Grounding scores (InfoNCE): s_i / τ

**Mathematical formulation**:
    z_i = CrossAttn(g, h_i, v_i)
    P(e_i | S_t) = softmax(W·z_i)
    P(a_t | S_t) = softmax(W_a · z_top)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Cross-Attention Block ────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block: query attends to key-value context.
    Pre-norm transformer convention with residual connections.
    No self-attention — simplified for stability.
    """

    def __init__(self, d_model: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,    # [B, Q, D]
        context: torch.Tensor,  # [B, C, D]
        context_mask: Optional[torch.Tensor] = None,  # [B, C] bool — True = ignore
    ) -> torch.Tensor:
        """Cross-attend query to context, then FFN. Returns [B, Q, D]."""
        # Cross-attention
        q = self.norm1(query)
        ca_out, _ = self.cross_attn(
            q, context, context,
            key_padding_mask=context_mask,
        )
        query = query + ca_out

        # FFN
        query = query + self.ffn(self.norm2(query))
        return query


# ── Unified VLA Grounder ─────────────────────────────────────

class VLAGrounder(nn.Module):
    """
    Unified grounding + action prediction module.

    Single cross-attention block fusing subgoal, GCN, and visual embeddings.
    Produces element logits, action logits, and grounding scores.

    Parameters
    ----------
    d_model : int
        Hidden dimension (256).
    text_dim : int
        Dimension of GCN node embeddings (256 after graph encoding).
    vision_dim : int
        Dimension of visual embeddings (512 for CLIP).
    subgoal_dim : int
        Dimension of subgoal text embeddings (384 for MiniLM).
    num_heads : int
        Number of attention heads (4).
    temperature : float
        InfoNCE temperature.
    dropout : float
        Dropout rate.
    num_action_types : int
        Number of action types (click, type, scroll, select).
    """

    def __init__(
        self,
        d_model: int = 256,
        text_dim: int = 256,
        vision_dim: int = 512,
        subgoal_dim: int = 384,
        num_heads: int = 4,
        temperature: float = 0.07,
        dropout: float = 0.1,
        num_action_types: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.num_action_types = num_action_types

        # Input projections into shared d_model space
        self.text_proj = nn.Linear(text_dim, d_model)
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.subgoal_proj = nn.Linear(subgoal_dim, d_model)

        # Single cross-attention block
        self.cross_attn = CrossAttentionBlock(d_model, num_heads, dropout)

        # Element logit head: z_i → scalar
        self.element_head = nn.Linear(d_model, 1)

        # Action logit head: z_top → [num_action_types]
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_action_types),
        )

        # Grounding score head for InfoNCE: [subgoal || node] → scalar
        self.score_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        subgoal_emb: torch.Tensor,     # [B, subgoal_dim]
        dom_node_embs: torch.Tensor,    # [B, N, text_dim]   (GCN-encoded)
        vision_embs: torch.Tensor,      # [B, N, vision_dim]
        node_mask: Optional[torch.Tensor] = None,  # [B, N] True = pad/ignore
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: grounding + element selection + action prediction.

        Returns
        -------
        element_logits  : [B, N] — logits for element selection
        action_logits   : [B, num_action_types] — action type logits
        grounding_scores: [B, N] — InfoNCE grounding scores
        node_reprs      : [B, N, d_model] — contextualized node representations
        """
        B, N = dom_node_embs.shape[:2]

        # Project to d_model
        sg = self.subgoal_proj(subgoal_emb).unsqueeze(1)  # [B, 1, D]
        dom = self.text_proj(dom_node_embs)                # [B, N, D]
        vis = self.vision_proj(vision_embs)                # [B, N, D]

        # Build multimodal context: [subgoal; DOM; vision]
        context = torch.cat([sg, dom, vis], dim=1)  # [B, 1+2N, D]

        # Build context mask (subgoal token never masked)
        if node_mask is not None:
            sg_mask = torch.zeros(B, 1, dtype=torch.bool, device=sg.device)
            context_mask = torch.cat([sg_mask, node_mask, node_mask], dim=1)  # [B, 1+2N]
        else:
            context_mask = None

        # Cross-attention: DOM nodes attend to full context
        query = dom  # [B, N, D]
        node_reprs = self.cross_attn(query, context, context_mask)
        node_reprs = self.output_norm(node_reprs)  # [B, N, D]

        # Element logits
        element_logits = self.element_head(node_reprs).squeeze(-1)  # [B, N]
        if node_mask is not None:
            element_logits = element_logits.masked_fill(node_mask, float("-inf"))

        # Grounding scores for InfoNCE
        sg_expanded = self.subgoal_proj(subgoal_emb).unsqueeze(1).expand(-1, N, -1)  # [B, N, D]
        paired = torch.cat([sg_expanded, node_reprs], dim=-1)  # [B, N, 2D]
        grounding_scores = self.score_head(paired).squeeze(-1)  # [B, N]
        if node_mask is not None:
            grounding_scores = grounding_scores.masked_fill(node_mask, float("-inf"))

        # Action logits from top-scoring element representation
        # Use attention-weighted mean of node representations
        elem_probs = F.softmax(element_logits.detach(), dim=-1)  # [B, N]
        if node_mask is not None:
            elem_probs = elem_probs.masked_fill(node_mask, 0.0)
        z_top = torch.bmm(elem_probs.unsqueeze(1), node_reprs).squeeze(1)  # [B, D]
        action_logits = self.action_head(z_top)  # [B, num_action_types]

        return element_logits, action_logits, grounding_scores, node_reprs

    def contrastive_loss(
        self,
        grounding_scores: torch.Tensor,  # [B, N]
        positive_indices: torch.Tensor,   # [B]
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss.

        L_InfoNCE = -log(exp(s_pos/τ) / Σ_j exp(s_j/τ))
        """
        logits = grounding_scores / self.temperature
        return F.cross_entropy(logits, positive_indices)

    def grounding_probabilities(
        self,
        scores: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:
        """Softmax probabilities over candidates."""
        return F.softmax(scores / self.temperature, dim=-1)


# ── Batched variant for variable-length DOM ──────────────────

class VLAGrounderBatched(nn.Module):
    """
    Handles variable-length DOM element sets by padding to max_N.
    Wraps :class:`VLAGrounder` with automatic padding/masking.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.grounder = VLAGrounder(**kwargs)
        self.d_model = self.grounder.d_model
        self.num_action_types = self.grounder.num_action_types

    def forward(
        self,
        subgoal_embs: torch.Tensor,        # [B, subgoal_dim]
        dom_node_embs_list: List[torch.Tensor],  # list of [N_i, text_dim]
        vision_embs_list: List[torch.Tensor],    # list of [N_i, vision_dim]
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Per-sample grounding with automatic padding.

        Returns
        -------
        all_elem_logits : list of [N_i] tensors (unpadded element logits)
        action_logits   : [B, num_action_types]
        all_scores      : list of [N_i] tensors (unpadded grounding scores)
        all_reprs       : list of [N_i, d_model] tensors (unpadded)
        """
        B = subgoal_embs.shape[0]
        device = subgoal_embs.device
        text_dim = dom_node_embs_list[0].shape[-1] if dom_node_embs_list[0].numel() > 0 else self.grounder.text_proj.in_features
        vis_dim = vision_embs_list[0].shape[-1] if vision_embs_list[0].numel() > 0 else self.grounder.vision_proj.in_features

        lengths = [e.shape[0] for e in dom_node_embs_list]
        max_N = max(lengths) if lengths else 1

        # Pad to max_N
        dom_padded = torch.zeros(B, max_N, text_dim, device=device)
        vis_padded = torch.zeros(B, max_N, vis_dim, device=device)
        mask = torch.ones(B, max_N, dtype=torch.bool, device=device)  # True = pad

        for i, n in enumerate(lengths):
            if n > 0:
                dom_padded[i, :n] = dom_node_embs_list[i]
                vis_padded[i, :n] = vision_embs_list[i]
                mask[i, :n] = False

        element_logits, action_logits, grounding_scores, node_reprs = self.grounder(
            subgoal_embs, dom_padded, vis_padded, mask
        )

        # Unpad
        all_elem_logits = [element_logits[i, :lengths[i]] for i in range(B)]
        all_scores = [grounding_scores[i, :lengths[i]] for i in range(B)]
        all_reprs = [node_reprs[i, :lengths[i]] for i in range(B)]

        return all_elem_logits, action_logits, all_scores, all_reprs

    def compute_loss_batched(
        self,
        subgoal_embs: torch.Tensor,
        dom_node_embs_list: List[torch.Tensor],
        vision_embs_list: List[torch.Tensor],
        positive_indices: List[int],
    ) -> torch.Tensor:
        """Mean InfoNCE contrastive loss across batch with padding."""
        B = subgoal_embs.shape[0]
        device = subgoal_embs.device
        text_dim = dom_node_embs_list[0].shape[-1] if dom_node_embs_list[0].numel() > 0 else self.grounder.text_proj.in_features
        vis_dim = vision_embs_list[0].shape[-1] if vision_embs_list[0].numel() > 0 else self.grounder.vision_proj.in_features

        lengths = [e.shape[0] for e in dom_node_embs_list]
        max_N = max(lengths) if lengths else 1

        dom_padded = torch.zeros(B, max_N, text_dim, device=device)
        vis_padded = torch.zeros(B, max_N, vis_dim, device=device)
        mask = torch.ones(B, max_N, dtype=torch.bool, device=device)

        valid_batch = []
        valid_targets = []

        for i, n in enumerate(lengths):
            if n > 0 and 0 <= positive_indices[i] < n:
                dom_padded[i, :n] = dom_node_embs_list[i]
                vis_padded[i, :n] = vision_embs_list[i]
                mask[i, :n] = False
                valid_batch.append(i)
                valid_targets.append(positive_indices[i])

        if not valid_batch:
            return torch.tensor(0.0, device=device, requires_grad=True)

        idx = torch.tensor(valid_batch, device=device)
        _, _, grounding_scores, _ = self.grounder(
            subgoal_embs[idx], dom_padded[idx], vis_padded[idx], mask[idx]
        )
        targets = torch.tensor(valid_targets, device=device, dtype=torch.long)
        return self.grounder.contrastive_loss(grounding_scores, targets)

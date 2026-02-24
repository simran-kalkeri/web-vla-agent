"""
Encoder modules for the VLA Web Agent (Refactored).

All encoders are frozen and produce fixed-dimensional vectors.

- TextEncoder   — DOM element / subgoal text → embedding (MiniLM, frozen)
- VisionEncoder — screenshot / crop → embedding (CLIP ViT-B/32, frozen)
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


# ── Text Encoder ─────────────────────────────────────────────

class TextEncoder(nn.Module):
    """
    Encode text strings into dense vectors using MiniLM (frozen).

    When *model_name* is ``None`` or unavailable, falls back to a
    lightweight bag-of-characters hash (useful for unit testing).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: int = 384,
        device: str = "cpu",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.device = device
        self._model = None
        self._tokenizer = None
        self.model_name = model_name

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=device)
            # Override output_dim to match model
            self.output_dim = self._model.get_sentence_embedding_dimension()
        except Exception:
            # Fallback: random projection
            self._proj = nn.Linear(256, output_dim)

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of strings.

        Returns
        -------
        Tensor of shape ``[len(texts), output_dim]``
        """
        if self._model is not None:
            with torch.no_grad():
                embs = self._model.encode(
                    texts, convert_to_tensor=True, show_progress_bar=False
                )
            return embs.to(self.device)

        # Fallback: deterministic hash-based embedding
        vecs = []
        for t in texts:
            codes = [ord(c) % 256 for c in t[:256]]
            codes += [0] * (256 - len(codes))
            vecs.append(codes)
        arr = torch.tensor(vecs, dtype=torch.float32, device=self.device) / 255.0
        return self._proj(arr)

    def encode_single(self, text: str) -> torch.Tensor:
        """Convenience for a single string → [1, dim]."""
        return self.forward([text])


# ── Vision Encoder ───────────────────────────────────────────

class VisionEncoder(nn.Module):
    """
    Encode images into dense vectors using CLIP ViT-B/32 (frozen).

    Falls back to a simple CNN when the CLIP model is unavailable.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        output_dim: int = 512,
        device: str = "cpu",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.device = device
        self._model = None
        self._processor = None
        self.model_name = model_name

        try:
            from transformers import CLIPModel, CLIPProcessor
            self._model = CLIPModel.from_pretrained(model_name).to(device)
            self._processor = CLIPProcessor.from_pretrained(model_name)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad = False
            self.output_dim = self._model.config.projection_dim
        except Exception:
            # Fallback: lightweight conv net
            self._fallback = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(32 * 4 * 4, output_dim),
            )

    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Encode a batch of PIL images.

        Returns
        -------
        Tensor of shape ``[len(images), output_dim]``
        """
        if self._model is not None and self._processor is not None:
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
            # L2-normalise
            return outputs / outputs.norm(dim=-1, keepdim=True)

        # Fallback
        tensors = []
        for img in images:
            arr = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr.transpose(2, 0, 1)))
        batch = torch.stack(tensors).to(self.device)
        return self._fallback(batch)

    def encode_single(self, image: Image.Image) -> torch.Tensor:
        """Convenience for a single image → [1, dim]."""
        return self.forward([image])

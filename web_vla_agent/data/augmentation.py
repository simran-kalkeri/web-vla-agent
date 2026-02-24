"""
Domain Perturbation & Augmentation for Generalization.

Implements data augmentation strategies to improve robustness
on unseen websites and domains.

**Research gap addressed**: Poor generalization to unseen domains.

Augmentations:
  1. DOM Structure Perturbation — random attribute dropping, node removal,
     class name shuffling
  2. Visual Augmentation — color jitter, random crop, blur on screenshots
  3. Text Masking — randomly mask element labels
  4. Contrastive Domain Regularization — perturbed vs. original should
     produce similar representations (consistency loss)
"""
from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from data.preprocessing import DOMElement


# ── Config ───────────────────────────────────────────────────

@dataclass
class AugmentationConfig:
    """Augmentation hyperparameters."""
    # DOM perturbation
    attribute_drop_rate: float = 0.15
    node_removal_rate: float = 0.05
    class_shuffle_rate: float = 0.10
    # Visual augmentation
    color_jitter_strength: float = 0.3
    random_crop_scale: Tuple[float, float] = (0.85, 1.0)
    blur_probability: float = 0.1
    blur_radius: float = 1.5
    # Text masking
    text_mask_rate: float = 0.30
    mask_token: str = "[MASK]"
    # Contrastive consistency
    consistency_weight: float = 0.1
    # Overall enable
    enabled: bool = True


# ── DOM Perturbation ─────────────────────────────────────────

class DOMPerturbation:
    """
    Apply random perturbations to DOM elements for domain robustness.

    Modifies elements in-place (operates on copies).
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()

    def __call__(self, elements: List[DOMElement]) -> List[DOMElement]:
        if not self.config.enabled:
            return elements

        # Deep copy to avoid modifying originals
        elements = [self._copy_element(e) for e in elements]

        # Random node removal (skip first element = root)
        if len(elements) > 2:
            elements = self._random_node_removal(elements)

        # Attribute dropping
        elements = [self._drop_attributes(e) for e in elements]

        # Class name shuffling
        elements = [self._shuffle_classes(e) for e in elements]

        return elements

    def _random_node_removal(self, elements: List[DOMElement]) -> List[DOMElement]:
        """Randomly remove some non-essential nodes."""
        keep = []
        for i, el in enumerate(elements):
            if i == 0 or random.random() > self.config.node_removal_rate:
                keep.append(el)
        return keep

    def _drop_attributes(self, el: DOMElement) -> DOMElement:
        """Randomly drop some attributes."""
        if not el.attributes:
            return el
        # Never drop 'id' or essential attrs
        protected = {"id", "type", "name", "href", "role"}
        new_attrs = {}
        for k, v in el.attributes.items():
            if k in protected or random.random() > self.config.attribute_drop_rate:
                new_attrs[k] = v
        el.attributes = new_attrs
        return el

    def _shuffle_classes(self, el: DOMElement) -> DOMElement:
        """Randomly shuffle CSS class names."""
        if "class" not in el.attributes or random.random() > self.config.class_shuffle_rate:
            return el
        classes = el.attributes["class"].split()
        random.shuffle(classes)
        el.attributes["class"] = " ".join(classes)
        return el

    @staticmethod
    def _copy_element(el: DOMElement) -> DOMElement:
        return DOMElement(
            element_id=el.element_id,
            tag=el.tag,
            text=el.text,
            attributes=dict(el.attributes),
            bounding_box=el.bounding_box,
            is_clickable=el.is_clickable,
            is_visible=el.is_visible,
            parent_id=el.parent_id,
            children_ids=list(el.children_ids),
            depth=el.depth,
        )


# ── Visual Augmentation ─────────────────────────────────────

class VisualAugmentation:
    """
    Apply random visual augmentations to screenshots.

    Augmentations are lightweight and designed for robustness,
    not photorealism.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()

    def __call__(self, image: Image.Image) -> Image.Image:
        if not self.config.enabled:
            return image

        image = image.copy()

        # Color jitter (brightness, contrast)
        image = self._color_jitter(image)

        # Random blur
        if random.random() < self.config.blur_probability:
            image = image.filter(
                ImageFilter.GaussianBlur(radius=self.config.blur_radius)
            )

        # Random crop + resize back
        image = self._random_crop(image)

        return image

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        """Simple colour jitter via numpy manipulation."""
        arr = np.asarray(image, dtype=np.float32)
        s = self.config.color_jitter_strength

        # Random brightness
        brightness = 1.0 + random.uniform(-s, s)
        arr = arr * brightness

        # Random contrast
        contrast = 1.0 + random.uniform(-s * 0.5, s * 0.5)
        mean = arr.mean()
        arr = (arr - mean) * contrast + mean

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _random_crop(self, image: Image.Image) -> Image.Image:
        """Crop a random sub-region and resize back."""
        w, h = image.size
        lo, hi = self.config.random_crop_scale
        scale = random.uniform(lo, hi)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w >= w and new_h >= h:
            return image
        x = random.randint(0, max(w - new_w, 0))
        y = random.randint(0, max(h - new_h, 0))
        cropped = image.crop((x, y, x + new_w, y + new_h))
        return cropped.resize((w, h), Image.LANCZOS)


# ── Text Masking ─────────────────────────────────────────────

class TextMasking:
    """
    Randomly mask element text labels to reduce reliance on
    specific text patterns.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()

    def __call__(self, elements: List[DOMElement]) -> List[DOMElement]:
        if not self.config.enabled:
            return elements

        result = []
        for el in elements:
            el_copy = DOMPerturbation._copy_element(el)
            if el_copy.text and random.random() < self.config.text_mask_rate:
                # Mask individual words
                words = el_copy.text.split()
                masked = [
                    self.config.mask_token if random.random() < 0.5 else w
                    for w in words
                ]
                el_copy.text = " ".join(masked)
            result.append(el_copy)
        return result


# ── Contrastive Consistency Loss ─────────────────────────────

def consistency_loss(
    original_embs: torch.Tensor,   # [N, D] — embeddings from original DOM
    perturbed_embs: torch.Tensor,  # [N, D] — embeddings from perturbed DOM
) -> torch.Tensor:
    """
    Contrastive domain regularization loss.

    Encourages the encoder to produce similar representations for
    original and perturbed DOM elements (same page, different surface).

    L_consist = 1 - mean(cos_sim(original_i, perturbed_i))
    """
    if original_embs.shape[0] == 0:
        return torch.tensor(0.0, device=original_embs.device, requires_grad=True)

    # Per-element cosine similarity
    cos_sim = F.cosine_similarity(original_embs, perturbed_embs, dim=-1)
    return 1.0 - cos_sim.mean()


# ── Unified Augmentation Pipeline ────────────────────────────

class AugmentationPipeline:
    """
    Apply all augmentations consistently to a training sample.

    Usage
    -----
    >>> aug = AugmentationPipeline()
    >>> elements_aug = aug.augment_dom(elements)
    >>> screenshot_aug = aug.augment_screenshot(screenshot)
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self.dom_perturbation = DOMPerturbation(self.config)
        self.visual_augmentation = VisualAugmentation(self.config)
        self.text_masking = TextMasking(self.config)

    def augment_dom(self, elements: List[DOMElement]) -> List[DOMElement]:
        """Apply DOM perturbation + text masking."""
        elements = self.dom_perturbation(elements)
        elements = self.text_masking(elements)
        return elements

    def augment_screenshot(self, image: Image.Image) -> Image.Image:
        """Apply visual augmentation."""
        return self.visual_augmentation(image)

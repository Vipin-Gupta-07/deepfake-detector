"""
model.py
--------
ViT-B/16 model definition and inference helper.

Architecture (from the paper):
  • Backbone  : ViT-B/16 pretrained on ImageNet-21k (via TIMM)
  • Input     : 224×224 RGB image split into 196 non-overlapping 16×16 patches
  • Embedding : 768-dim projection per patch + learnable CLS token
  • Encoder   : 12 Transformer layers, 12 attention heads
  • Head      : Linear(768, 1) → Sigmoid  (binary: Real / Fake)
"""

import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)

# Path where the fine-tuned weights should be placed
DEFAULT_WEIGHTS_PATH = Path(__file__).parent / "models" / "best_model.pth"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DeepfakeViT(nn.Module):
    """
    Thin wrapper around TIMM's vit_base_patch16_224 that replaces the
    default multi-class head with a single sigmoid-activated output for
    binary (Real / Fake) classification.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load base ViT-B/16 (no head) —  pretrained=True pulls ImageNet-21k weights
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,          # removes default classifier head
        )

        # Custom binary classification head
        # hidden_dim = 768  (ViT-B projection dimension)
        hidden_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 224, 224) normalised float32 tensor

        Returns
        -------
        logits : (B, 1) probability in [0, 1];  > 0.5 → Fake
        """
        features = self.backbone(x)   # (B, 768)  — CLS token embedding
        return self.head(features)     # (B, 1)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_model(
    weights_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> DeepfakeViT:
    """
    Instantiate and return a DeepfakeViT model.

    If *weights_path* points to an existing file, fine-tuned weights are
    loaded from it.  Otherwise, the model is returned with ImageNet-21k
    pretrained backbone weights only (useful for development / testing
    without a .pth file).

    Parameters
    ----------
    weights_path : path to ``best_model.pth``; defaults to
                   ``./models/best_model.pth`` relative to this file
    device       : torch device; auto-detected when None
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = weights_path or DEFAULT_WEIGHTS_PATH

    # Build model with pretrained backbone
    model = DeepfakeViT(pretrained=True)

    # Attempt to load fine-tuned weights
    if weights_path.exists():
        logger.info("Loading fine-tuned weights from: %s", weights_path)
        state = torch.load(weights_path, map_location=device)
        # Support both raw state_dict and checkpoints that wrap it
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        logger.info("Fine-tuned weights loaded successfully.")
    else:
        logger.warning(
            "Fine-tuned weights NOT found at '%s'. "
            "Running with ImageNet pretrained backbone only "
            "(predictions will NOT be accurate until you place best_model.pth).",
            weights_path,
        )

    model.to(device)
    model.eval()
    logger.info("Model ready on device=%s", device)
    return model


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

THRESHOLD = 0.5   # probability threshold; > THRESHOLD → Fake


@torch.inference_mode()
def predict(
    model: DeepfakeViT,
    tensor: torch.Tensor,
    device: torch.device,
    threshold: float = THRESHOLD,
) -> dict:
    """
    Run a single-image inference pass.

    Parameters
    ----------
    model     : loaded DeepfakeViT
    tensor    : preprocessed (1, 3, 224, 224) float32 tensor
    device    : torch device
    threshold : decision boundary

    Returns
    -------
    dict with keys:
        label       – "Fake" | "Real"
        confidence  – float in [0, 1]; probability of *Fake*
        real_prob   – probability of *Real*
        fake_prob   – probability of *Fake*
    """
    tensor = tensor.to(device)
    prob_fake: float = model(tensor).squeeze().item()   # scalar in [0,1]
    prob_real: float = 1.0 - prob_fake

    label = "Fake" if prob_fake >= threshold else "Real"
    confidence = prob_fake if label == "Fake" else prob_real

    return {
        "label":      label,
        "confidence": round(confidence * 100, 2),   # as percentage
        "real_prob":  round(prob_real * 100, 2),
        "fake_prob":  round(prob_fake * 100, 2),
    }

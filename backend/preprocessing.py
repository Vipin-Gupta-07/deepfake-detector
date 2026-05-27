"""
preprocessing.py
----------------
Face detection and image preprocessing pipeline.

Pipeline steps (as per the research paper):
  1. Detect face with MTCNN (facenet-pytorch).
  2. Crop face region + add a small border.
  3. Resize to 224×224.
  4. Fallback: resize full image if no face is detected.
  5. Normalise with ImageNet statistics.
"""

import io
import logging
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ImageNet normalisation constants (required by ViT-B/16 pretrained on 21k)
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Final transform applied after cropping/resizing
_NORMALISE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


class FacePreprocessor:
    """
    Wraps MTCNN face detection and produces a normalised 224×224 tensor
    ready for ViT-B/16 inference.
    """

    def __init__(
        self,
        target_size: int = 224,
        border_fraction: float = 0.10,
        device: Optional[torch.device] = None,
    ):
        """
        Parameters
        ----------
        target_size      : output spatial dimension (square)
        border_fraction  : extra border around the detected face bounding box
                           expressed as a fraction of the box side (default 10 %)
        device           : torch device; auto-detected when None
        """
        self.target_size      = target_size
        self.border_fraction  = border_fraction
        self.device           = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # MTCNN – keep_all=False returns the single most-prominent face
        self.mtcnn = MTCNN(
            image_size=target_size,
            keep_all=False,
            device=self.device,
            post_process=False,   # we handle normalisation ourselves
            min_face_size=80,     # optimized: skip searching for tiny faces
        )

        logger.info(
            "FacePreprocessor initialised on device=%s, target_size=%d",
            self.device,
            target_size,
        )

    # ------------------------------------------------------------------
    def _add_border(
        self,
        img: Image.Image,
        box: list[float],
    ) -> Image.Image:
        """Crop image to *box* with an extra border, clamped to image bounds."""
        w, h = img.size
        x1, y1, x2, y2 = box

        # Compute border in pixels
        side   = max(x2 - x1, y2 - y1)
        border = int(side * self.border_fraction)

        x1 = max(0, int(x1) - border)
        y1 = max(0, int(y1) - border)
        x2 = min(w, int(x2) + border)
        y2 = min(h, int(y2) + border)

        return img.crop((x1, y1, x2, y2))

    # ------------------------------------------------------------------
    def process(self, image_bytes: bytes) -> tuple[torch.Tensor, bool]:
        """
        Convert raw image bytes into a normalised (1, 3, 224, 224) tensor.

        Returns
        -------
        tensor      : float32 tensor, shape (1, 3, 224, 224)
        face_found  : True if MTCNN detected a face, False if fallback was used
        """
        # ---- Load image ----
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ---- Attempt face detection ----
        face_found = False
        try:
            boxes, _ = self.mtcnn.detect(img)
            if boxes is not None and len(boxes) > 0:
                # Use the first (most prominent) detected face
                cropped = self._add_border(img, boxes[0])
                face_found = True
                logger.debug("MTCNN detected face; box=%s", boxes[0])
            else:
                logger.debug("MTCNN found no face; using full image as fallback")
                cropped = img
        except Exception as exc:
            logger.warning("MTCNN error (%s); using full image as fallback", exc)
            cropped = img

        # ---- Resize and normalise ----
        resized = cropped.resize(
            (self.target_size, self.target_size), Image.LANCZOS
        )
        tensor = _NORMALISE(resized).unsqueeze(0)  # → (1, 3, 224, 224)

        return tensor, face_found

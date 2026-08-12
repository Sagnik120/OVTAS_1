"""
CLIP encoder backend.

CLIP (Radford et al., 2021) trains its dual towers with the symmetric
InfoNCE contrastive loss over image-text pairs in a batch. The paper's
empirical study (Table III) finds CLIP-family models slightly behind
SigLIP on average TAS score, but still a strong, widely-available
choice.

This module is intentionally the *only* place that imports
``torch`` / ``open_clip`` -- everything else in ``ovtas`` works with
plain NumPy arrays, so the rest of the pipeline is testable without
these (comparatively heavy) dependencies installed. Install them with:

    pip install -e ".[vlm]"

Recommended checkpoint for limited hardware (e.g. a single consumer
GPU or CPU-only laptop): ``ViT-B-32`` / ``openai`` pretraining
(~150M parameters, ~600MB), which is the default below.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ovtas.encoders.base import BaseVLMEncoder, _l2_normalize_rows
from ovtas.encoders.registry import ENCODERS


@ENCODERS.register("clip")
class ClipEncoder(BaseVLMEncoder):
    """OpenAI CLIP via the ``open_clip`` library.

    Parameters
    ----------
    model_name:
        An ``open_clip`` architecture name, e.g. ``"ViT-B-32"``
        (default; smallest common variant, good for limited hardware)
        or ``"ViT-L-14"`` (larger, higher quality, needs more VRAM).
    pretrained:
        Pretraining tag understood by ``open_clip``, e.g. ``"openai"``.
    device:
        ``"cpu"`` or ``"cuda"``. Defaults to CUDA if available.
    batch_size:
        Batch size used internally when encoding many images at once.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = None,
        batch_size: int = 32,
    ):
        try:
            import open_clip  # noqa: F401
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only w/o extras
            raise ImportError(
                "ClipEncoder requires the optional 'vlm' extra. Install it with:\n"
                "    pip install -e \".[vlm]\"\n"
                "(This pulls in torch, open-clip-torch, and Pillow.)"
            ) from exc

        self._torch = torch
        self.name = f"clip-{model_name}-{pretrained}"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device).eval()
        self._embed_dim = self.model.visual.output_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode_images(self, images: Sequence) -> np.ndarray:
        torch = self._torch
        all_feats = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                tensors = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
                feats = self.model.encode_image(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_feats.append(feats.cpu().numpy())
        if not all_feats:
            return np.zeros((0, self._embed_dim))
        return _l2_normalize_rows(np.concatenate(all_feats, axis=0))

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            tokens = self.tokenizer(list(texts)).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return _l2_normalize_rows(feats.cpu().numpy())

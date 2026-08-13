"""
SigLIP encoder backend.

SigLIP (Zhai et al., 2023) replaces CLIP's softmax InfoNCE loss with
an independent sigmoid binary loss per image-text pair, removing the
need for a global normalization across the batch. The paper's
empirical study (Table III) finds SigLIP to be the strongest VLM
family for zero-shot TAS on average, making it the recommended default
backbone for this codebase.

Like ``clip_encoder.py``, this module is the only place that imports
``torch`` / ``transformers``. Install the optional dependencies with:

    pip install -e ".[vlm]"

Recommended checkpoint for limited hardware: ``google/siglip-base
-patch16-224`` (~203M parameters), the default below.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ovtas.encoders.base import BaseVLMEncoder, _l2_normalize_rows
from ovtas.encoders.registry import ENCODERS


@ENCODERS.register("siglip")
class SiglipEncoder(BaseVLMEncoder):
    """SigLIP via HuggingFace ``transformers``.

    Parameters
    ----------
    model_name:
        A HuggingFace Hub checkpoint id, e.g.
        ``"google/siglip-base-patch16-224"`` (default).
    device:
        ``"cpu"`` or ``"cuda"``. Defaults to CUDA if available.
    batch_size:
        Batch size used internally when encoding many images at once.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str = None,
        batch_size: int = 32,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - exercised only w/o extras
            raise ImportError(
                "SiglipEncoder requires the optional 'vlm' extra. Install it with:\n"
                "    pip install -e \".[vlm]\"\n"
                "(This pulls in torch, transformers, and Pillow.)"
            ) from exc

        self._torch = torch
        self.name = f"siglip-{model_name.split('/')[-1]}"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        # FRAGILITY NOTE: text_config and vision_config should have the same
        # hidden_size for SigLIP, but this is not guaranteed for all checkpoints.
        # Revisit if ever switching to an asymmetric SigLIP variant.
        self._embed_dim = self.model.config.text_config.hidden_size

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode_images(self, images: Sequence) -> np.ndarray:
        torch = self._torch
        all_feats = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                inputs = self.processor(images=list(batch), return_tensors="pt").to(self.device)
                feats = self.model.get_image_features(**inputs)
                # get_image_features() returns BaseModelOutputWithPooling in
                # this transformers version; the embedding tensor is .pooler_output.
                all_feats.append(feats.pooler_output.cpu().numpy())
        if not all_feats:
            return np.zeros((0, self._embed_dim))
        return _l2_normalize_rows(np.concatenate(all_feats, axis=0))

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            inputs = self.processor(
                text=list(texts), return_tensors="pt", padding="max_length"
            ).to(self.device)
            feats = self.model.get_text_features(**inputs)
        # get_text_features() returns BaseModelOutputWithPooling in this
        # transformers version; the embedding tensor is .pooler_output.
        return _l2_normalize_rows(feats.pooler_output.cpu().numpy())

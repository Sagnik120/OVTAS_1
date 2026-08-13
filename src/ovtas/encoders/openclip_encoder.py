"""
OpenCLIP encoder backend.

OpenCLIP (Cherti et al., 2023) reproduces CLIP training at scale on
public datasets (LAION-400M, LAION-2B, DataComp). The paper (Table III)
benchmarks several OpenCLIP checkpoints alongside OpenAI CLIP and SigLIP.

The config key is ``"openclip"``. The two fields that matter are:
    model_name  -- OpenCLIP architecture, e.g. "ViT-B-16"
    pretrained  -- OpenCLIP dataset tag, e.g. "laion2b_s34b_b88k"

Install with:
    pip install -e ".[vlm]"
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ovtas.encoders.base import BaseVLMEncoder, _l2_normalize_rows, resolve_torch_device
from ovtas.encoders.registry import ENCODERS


@ENCODERS.register("openclip")
class OpenCLIPEncoder(BaseVLMEncoder):
    """OpenCLIP via the ``open_clip`` library.

    Parameters
    ----------
    model_name:
        OpenCLIP architecture tag, e.g. ``"ViT-B-16"`` or ``"ViT-L-14"``.
    pretrained:
        OpenCLIP pretraining dataset tag, e.g. ``"laion2b_s34b_b88k"``.
        See ``open_clip.list_pretrained()`` for available combinations.
    device:
        ``"cpu"``, ``"cuda"``, or ``"mps"``. Auto-detected if omitted.
    batch_size:
        Number of images processed per forward pass.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        device: str = None,
        batch_size: int = 32,
    ):
        try:
            import open_clip  # noqa: F401
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenCLIPEncoder requires the optional 'vlm' extra. Install it with:\n"
                "    pip install -e \".[vlm]\"\n"
                "(This pulls in torch, open-clip-torch, and Pillow.)"
            ) from exc

        self._torch = torch
        self.name = f"openclip-{model_name}-{pretrained}"
        self.device = resolve_torch_device(device)
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

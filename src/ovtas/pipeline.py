"""
End-to-end OVTAS pipeline: encoder -> Stage 1 (FAES) -> Stage 2 (SMTS).

This module is deliberately thin: all of the real logic already lives
in well-tested, independent modules (``ovtas.encoders``,
``ovtas.stage1_faes``, ``ovtas.stage2_smts``). :class:`OVTASPipeline`
just wires them together in the order the paper specifies, using the
:mod:`ovtas.registry` lookups so any registered encoder / prompt
builder can be selected purely by name/config, with zero code changes
here when a new one is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ovtas.encoders import ENCODERS, BaseVLMEncoder
from ovtas.stage1_faes import (
    FAESOutput,
    FAESPlusConfig,
    FAESPlusOutput,
    build_prompts,
    compute_faes,
    compute_faes_plus,
)
from ovtas.stage2_smts import ASOTConfig, SMTSOutput, decode_asot, shuffle_action_order


@dataclass
class OVTASResult:
    """Everything produced by one end-to-end pipeline run, for a
    single video, kept together for downstream evaluation/plotting.
    """

    video_id: Optional[str]
    faes: FAESOutput
    smts: SMTSOutput
    label_names: List[str]
    action_order: np.ndarray  # permutation applied before decoding
    faes_plus: Optional[FAESPlusOutput] = None

    @property
    def predicted_labels(self) -> np.ndarray:
        """Per-frame predicted label indices, mapped back to the
        *original* (un-shuffled) action-index space.
        """
        # smts.labels indexes into the shuffled columns; invert the
        # permutation to recover original action ids.
        return self.action_order[self.smts.labels]

    @property
    def predicted_label_names(self) -> List[str]:
        return [self.label_names[i] for i in self.predicted_labels]


class OVTASPipeline:
    """Training-free, zero-shot, open-vocabulary TAS pipeline.

    Supports both baseline 2-stage OVTAS and proposed 3-stage OVTAS+ (with FAES+).

    Parameters
    ----------
    encoder:
        A pre-built :class:`~ovtas.encoders.base.BaseVLMEncoder`
        instance. Alternatively, pass ``encoder_name`` (+ kwargs) to
        have the pipeline build one from the registry.
    encoder_name / encoder_kwargs:
        Used only if ``encoder`` is not given directly.
    faes_plus_config:
        Hyperparameters for the proposed Stage 2 FAES+ attention refinement.
        If None or ``enabled=False``, baseline raw FAES is used.
    asot_config:
        Hyperparameters for the Stage 2/3 decoder. Defaults to the
        paper's fixed values (see :class:`ASOTConfig`).
    prompt_dataset:
        Which registered prompt-builder to use for turning raw action
        labels into natural-language phrases (see
        ``ovtas.stage1_faes.prompts``). Use ``"gtea"`` for GTEA-style
        ``(verb, nouns)`` label tuples, ``"default"`` otherwise.
    shuffle_actions:
        If True (default, matching the paper's action-set supervision
        setting), randomize the action-label order before building the
        temporal prior, so no positional bias about label order leaks
        into Stage 2/3.
    seed:
        RNG seed controlling the action-order shuffle, for
        reproducibility.
    """

    def __init__(
        self,
        encoder: Optional[BaseVLMEncoder] = None,
        encoder_name: str = "mock",
        encoder_kwargs: Optional[dict] = None,
        faes_plus_config: Optional[FAESPlusConfig] = None,
        asot_config: Optional[ASOTConfig] = None,
        prompt_dataset: str = "default",
        shuffle_actions: bool = True,
        seed: int = 0,
    ):
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = ENCODERS.build(encoder_name, **(encoder_kwargs or {}))

        self.faes_plus_config = (
            faes_plus_config if faes_plus_config is not None else FAESPlusConfig(enabled=False)
        )
        self.asot_config = asot_config or ASOTConfig()
        self.prompt_dataset = prompt_dataset
        self.shuffle_actions = shuffle_actions
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        frames: Sequence,
        action_labels: Sequence,
        video_id: Optional[str] = None,
    ) -> OVTASResult:
        """Run the full pipeline on one video.

        Parameters
        ----------
        frames:
            Sequence of frame images (whatever the chosen encoder's
            ``encode_images`` accepts -- e.g. PIL Images for CLIP /
            SigLIP, or raw ``np.ndarray`` for :class:`MockEncoder`).
        action_labels:
            The dataset's open-vocabulary candidate action set (raw
            tokens, or ``(verb, nouns)`` tuples for the "gtea" prompt
            builder). This is the *set* of possible actions, not
            necessarily all present in this particular video.
        video_id:
            Optional identifier, carried through into the result for
            bookkeeping/logging.

        Returns
        -------
        OVTASResult
        """
        label_names = list(action_labels)
        prompts = build_prompts(label_names, dataset=self.prompt_dataset)

        frame_embeddings = self.encoder.encode_images(list(frames))
        action_embeddings = self.encoder.encode_text(prompts)

        if self.faes_plus_config.enabled:
            faes_out = compute_faes_plus(
                frame_embeddings,
                action_embeddings,
                config=self.faes_plus_config,
                already_normalized=True,
            )
        else:
            faes_out = compute_faes(
                frame_embeddings, action_embeddings, already_normalized=True
            )

        num_actions = faes_out.num_actions
        if self.shuffle_actions:
            action_order = shuffle_action_order(num_actions, rng=self.rng)
        else:
            action_order = np.arange(num_actions)

        # Apply the permutation to the similarity matrix's columns
        # before decoding, so the temporal prior R's column ordering
        # carries no leaked positional information about label order.
        shuffled_similarity = faes_out.similarity[:, action_order]
        smts_out = decode_asot(shuffled_similarity, config=self.asot_config)

        return OVTASResult(
            video_id=video_id,
            faes=faes_out,
            smts=smts_out,
            label_names=label_names,
            action_order=action_order,
            faes_plus=faes_out if isinstance(faes_out, FAESPlusOutput) else None,
        )


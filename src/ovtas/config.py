"""
Dataclass-based configuration for the OVTAS pipeline.

Using plain dataclasses (rather than a dict passed around everywhere)
gives IDE autocompletion, type checking, and a single place that
documents every knob. :func:`load_config` reads a YAML file (see
``configs/*.yaml``) into a :class:`OVTASConfig`; :func:`save_config`
writes one back out, useful for logging the exact config a run used.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import yaml

from ovtas.stage1_faes.faes_plus import FAESPlusConfig
from ovtas.stage2_smts.asot_decoder import ASOTConfig


@dataclass
class EncoderConfig:
    """Which VLM backbone to use and how to run it."""

    #: One of the names registered in ``ovtas.encoders.ENCODERS``
    #: (e.g. "mock", "clip", "siglip").
    name: str = "mock"
    model_name: Optional[str] = None
    pretrained: Optional[str] = None
    device: Optional[str] = None
    batch_size: int = 32
    #: Extra dim only used by MockEncoder; ignored by real encoders.
    embed_dim: int = 32


@dataclass
class DataConfig:
    """Which dataset to use and how to sample frames from it."""

    #: One of the names registered in ``ovtas.data.DATASETS``
    #: (e.g. "synthetic", "gtea").
    name: str = "synthetic"
    root: Optional[str] = None
    #: Only every ``frame_stride``-th frame is encoded, trading
    #: temporal resolution for speed on constrained hardware.
    frame_stride: int = 1
    max_frames: Optional[int] = None
    #: Prompt-builder key from ``ovtas.stage1_faes.PROMPT_BUILDERS``.
    prompt_dataset: str = "default"


@dataclass
class OVTASConfig:
    """Top-level configuration bundling every stage's settings."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    data: DataConfig = field(default_factory=DataConfig)
    #: Stage 2 FAES+ attention refinement settings (OVTAS+).
    faes_plus: FAESPlusConfig = field(default_factory=lambda: FAESPlusConfig(enabled=False))
    #: Stage 3 SMTS Optimal Transport decoder settings.
    asot: ASOTConfig = field(default_factory=ASOTConfig)
    #: Randomize the action-column order before decoding, per the
    #: paper's action-set (not action-*sequence*) supervision setting.
    shuffle_actions: bool = True
    seed: int = 0
    output_dir: str = "outputs"


def _dict_to_dataclass(cls, data: dict):
    """Recursively build a (possibly nested) dataclass from a dict,
    ignoring unknown keys so configs can be forward-compatible with
    older code and vice-versa.
    """
    if data is None:
        return cls()
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            continue
        field_type = field_types[key]
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(field_type, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str) -> OVTASConfig:
    """Load an :class:`OVTASConfig` from a YAML file."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _dict_to_dataclass(OVTASConfig, raw)


def save_config(config: OVTASConfig, path: str) -> None:
    """Write an :class:`OVTASConfig` back out to YAML."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(dataclasses.asdict(config), fh, sort_keys=False)

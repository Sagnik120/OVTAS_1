"""Unit tests for ovtas.config (dataclass schema, validation, YAML serialization)."""

from __future__ import annotations

import os
import pytest

from ovtas.config import DataConfig, EncoderConfig, OVTASConfig, load_config, save_config
from ovtas.stage1_faes import FAESPlusConfig
from ovtas.stage2_smts import ASOTConfig


@pytest.mark.unit
def test_default_config_values():
    cfg = OVTASConfig()
    assert cfg.encoder.name == "mock"
    assert cfg.data.name == "synthetic"
    assert cfg.asot.epsilon == 0.07
    assert cfg.shuffle_actions is True
    assert not cfg.faes_plus.enabled


@pytest.mark.unit
def test_save_and_load_round_trip(tmp_path):
    cfg = OVTASConfig(
        encoder=EncoderConfig(name="siglip", model_name="google/siglip-base-patch16-224"),
        data=DataConfig(name="gtea", root="/data/gtea", frame_stride=5),
        faes_plus=FAESPlusConfig(enabled=True, tau=0.08, beta=0.6, temporal_window=5),
        asot=ASOTConfig(epsilon=0.05, rho=0.4),
        seed=123,
    )
    path = str(tmp_path / "config.yaml")
    save_config(cfg, path)
    assert os.path.isfile(path)

    loaded = load_config(path)
    assert loaded.encoder.name == "siglip"
    assert loaded.encoder.model_name == "google/siglip-base-patch16-224"
    assert loaded.data.name == "gtea"
    assert loaded.data.root == "/data/gtea"
    assert loaded.data.frame_stride == 5
    assert loaded.seed == 123
    assert loaded.faes_plus.enabled is True
    assert abs(loaded.faes_plus.tau - 0.08) < 1e-6
    assert abs(loaded.faes_plus.beta - 0.6) < 1e-6
    assert loaded.faes_plus.temporal_window == 5
    assert loaded.asot.epsilon == 0.05
    assert loaded.asot.rho == 0.4


@pytest.mark.unit
def test_load_config_ignores_unknown_keys(tmp_path):
    path = str(tmp_path / "config.yaml")
    with open(path, "w") as fh:
        fh.write("seed: 7\nsome_future_field: 42\n")
    cfg = load_config(path)
    assert cfg.seed == 7


@pytest.mark.unit
def test_load_config_handles_empty_file(tmp_path):
    path = str(tmp_path / "empty.yaml")
    open(path, "w").close()
    cfg = load_config(path)
    assert cfg == OVTASConfig()

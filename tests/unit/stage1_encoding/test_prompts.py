"""Unit tests for Stage 1 prompt construction."""

from __future__ import annotations

import pytest

from ovtas.stage1_faes.prompts import (
    PROMPT_BUILDERS,
    PromptTemplate,
    build_prompts,
    normalize_label,
)


@pytest.mark.unit
def test_normalize_label_basic():
    assert normalize_label("pour_coffee") == "pour coffee"
    assert normalize_label("cut-tomato") == "cut tomato"
    assert normalize_label("  add_sugar  ") == "add sugar"


@pytest.mark.unit
def test_normalize_label_background_aliases():
    for raw in ["SIL", "sil", "background", "NONE"]:
        assert normalize_label(raw) == "background"


@pytest.mark.unit
def test_build_prompts_default_dataset():
    labels = ["pour_coffee", "add_sugar", "SIL"]
    prompts = build_prompts(labels, dataset="default")
    assert prompts == ["pour coffee", "add sugar", "background"]


@pytest.mark.unit
def test_build_prompts_gtea_dataset_with_verb_noun_tuples():
    labels = [("pour", ["coffee"]), ("take", ["cup", "plate"])]
    prompts = build_prompts(labels, dataset="gtea")
    assert prompts == ["pour coffee", "take cup plate"]


@pytest.mark.unit
def test_build_prompts_unknown_dataset_raises():
    with pytest.raises(KeyError):
        build_prompts(["a"], dataset="does_not_exist")

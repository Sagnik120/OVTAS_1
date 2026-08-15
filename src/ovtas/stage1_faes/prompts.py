"""
Prompt construction for Stage 1 (FAES).

The paper describes prompt construction as: "Each action label is
normalized into a natural-language phrase (e.g., 'pour_coffee' ->
'pour coffee'). These normalized labels are tokenized and encoded by
the VLM text encoder." (Sec. III-D.1)

For GTEA specifically the paper notes: "For GTEA dataset we attach
verbs to construct short phrases using the annotation." GTEA action
labels are commonly stored as a (verb, noun-list) pair, e.g.
verb="pour", nouns=["coffee"] which we join into "pour coffee".

This module keeps that normalization logic isolated and testable so
new datasets can add their own normalization rule without touching
the encoder or similarity code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from ovtas.registry import Registry

# Registry so a new dataset can plug in its own label -> prompt rule
# via `@PROMPT_BUILDERS.register("my_dataset")` without editing this
# file.
PROMPT_BUILDERS: Registry = Registry("prompt_builder")


def normalize_label(label: str) -> str:
    """Turn a raw action-label token into a natural language phrase.

    Examples
    --------
    >>> normalize_label("pour_coffee")
    'pour coffee'
    >>> normalize_label("SIL")
    'background'
    >>> normalize_label("cut-tomato")
    'cut tomato'
    >>> normalize_label("  add_sugar  ")
    'add sugar'
    """
    text = label.strip()
    if text.upper() in {"SIL", "BACKGROUND", "NONE", "__BACKGROUND__"}:
        return "background"
    # Replace underscores/hyphens with spaces, collapse camelCase,
    # then collapse repeated whitespace.
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)  # camelCase split
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


@dataclass(frozen=True)
class PromptTemplate:
    """A simple, composable template of the form ``prefix + phrase``.

    Using a template object (instead of raw f-strings scattered around
    the codebase) makes it trivial to add new prompting strategies
    later (e.g. ensembling several templates, as CLIP's original
    zero-shot recipe does) without touching FAES.
    """

    prefix: str = ""
    suffix: str = ""

    def render(self, phrase: str) -> str:
        return f"{self.prefix}{phrase}{self.suffix}".strip()


DEFAULT_TEMPLATE = PromptTemplate(prefix="", suffix="")


@PROMPT_BUILDERS.register("default")
class DefaultPromptBuilder:
    """Generic normalize-and-template prompt builder used for most
    datasets (e.g. Breakfast, 50 Salads) whose labels are already
    single verb-object tokens like ``pour_coffee``.
    """

    def __init__(self, template: PromptTemplate = DEFAULT_TEMPLATE) -> None:
        self.template = template

    def build(self, labels: Sequence[str]) -> List[str]:
        return [self.template.render(normalize_label(lbl)) for lbl in labels]


@PROMPT_BUILDERS.register("gtea")
class GTEAPromptBuilder:
    """GTEA-specific builder.

    GTEA annotations decompose an action into a verb and one or more
    object nouns (e.g. verb="pour", nouns=["coffee"]). This builder
    accepts either plain strings (already-joined labels, normalized
    the default way) or ``(verb, nouns)`` tuples and joins them into a
    short phrase, matching the paper's description of "attaching
    verbs to construct short phrases" for GTEA.
    """

    def __init__(self, template: PromptTemplate = DEFAULT_TEMPLATE) -> None:
        self.template = template

    def build(self, labels: Sequence) -> List[str]:
        phrases: List[str] = []
        for lbl in labels:
            if isinstance(lbl, (tuple, list)) and len(lbl) == 2:
                verb, nouns = lbl
                if isinstance(nouns, str):
                    nouns = [nouns]
                phrase = " ".join([verb, *nouns])
            else:
                phrase = str(lbl)
            phrases.append(self.template.render(normalize_label(phrase)))
        return phrases


def build_prompts(
    labels: Iterable,
    dataset: str = "default",
    template: Optional[PromptTemplate] = None,
) -> List[str]:
    """Convenience function: look up the right builder and run it.

    Parameters
    ----------
    labels:
        Raw action-label tokens (or ``(verb, nouns)`` pairs for GTEA).
    dataset:
        Name of a registered prompt builder, e.g. "default" or "gtea".
    template:
        Optional custom :class:`PromptTemplate`. If omitted, the
        builder's default template is used.
    """
    kwargs = {} if template is None else {"template": template}
    builder = PROMPT_BUILDERS.build(dataset, **kwargs)
    return builder.build(list(labels))

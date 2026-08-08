"""
Generic registry pattern used throughout OVTAS.

Every extensible family of components (VLM encoders, OT solvers,
datasets, baselines, ...) owns one ``Registry`` instance. New
implementations register themselves with a decorator instead of
requiring edits to a central ``if/elif`` chain, which is what makes it
easy to bolt on a new VLM, a new loss, or a new dataset later without
touching existing, tested code.

Example
-------
>>> from ovtas.registry import Registry
>>> ENCODERS = Registry("encoder")
>>>
>>> @ENCODERS.register("dummy")
... class DummyEncoder:
...     pass
>>>
>>> ENCODERS.get("dummy") is DummyEncoder
True
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterator, Type, TypeVar

T = TypeVar("T")


class RegistryError(KeyError):
    """Raised for duplicate registrations or unknown lookups."""


class Registry(Generic[T]):
    """A minimal, dependency-free name -> class/factory registry.

    Parameters
    ----------
    kind:
        Human readable name of the family this registry manages
        (used only for clearer error messages, e.g. "encoder").
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: Dict[str, Type[T]] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Class decorator that registers ``name -> cls``.

        Raises
        ------
        RegistryError
            If ``name`` is already registered, to catch accidental
            duplicate names early (e.g. copy-pasted decorators).
        """

        def _decorator(cls: Type[T]) -> Type[T]:
            key = name.lower()
            if key in self._entries:
                raise RegistryError(
                    f"{self._kind} '{name}' is already registered to "
                    f"{self._entries[key]!r}; choose a unique name."
                )
            self._entries[key] = cls
            return cls

        return _decorator

    def get(self, name: str) -> Type[T]:
        key = name.lower()
        if key not in self._entries:
            available = ", ".join(sorted(self._entries)) or "<empty>"
            raise RegistryError(
                f"Unknown {self._kind} '{name}'. Available: {available}"
            )
        return self._entries[key]

    def build(self, name: str, *args, **kwargs) -> T:
        """Instantiate the registered class for ``name``."""
        return self.get(name)(*args, **kwargs)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._entries))

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Registry(kind={self._kind!r}, entries={sorted(self._entries)})"

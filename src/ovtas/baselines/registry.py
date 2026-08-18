"""Shared registry instance for training-free baselines.

Kept in its own tiny module (rather than inside ``__init__.py``) so
that ``random_uniform.py`` and ``equal_splits.py`` can both import
``BASELINES`` without creating a circular import through the package
``__init__``.
"""

from ovtas.registry import Registry

BASELINES: Registry = Registry("baseline")

"""Shared registry instance for Vision-Language Model encoders.

Kept in its own module (mirroring ``ovtas.baselines.registry``) so
individual encoder implementations can register themselves without a
circular import through the package ``__init__``.
"""

from ovtas.registry import Registry

ENCODERS: Registry = Registry("encoder")

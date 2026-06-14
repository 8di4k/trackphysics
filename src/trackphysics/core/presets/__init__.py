"""Registry of generic physics presets (BRIEF.md §8).

Importing this package registers the built-in generic presets (currently ``sphere``),
so ``get_preset("sphere")`` works after ``import trackphysics``.
"""

from __future__ import annotations

from .base import (
    EventDetector,
    PhysicsPreset,
    get_preset,
    list_presets,
    register_preset,
)
from .sphere import SpherePreset  # noqa: E402  (registers the "sphere" preset on import)

__all__ = [
    "EventDetector",
    "PhysicsPreset",
    "SpherePreset",
    "get_preset",
    "list_presets",
    "register_preset",
]

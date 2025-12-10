"""Detector plugin registry.

Allows registering detector factories by name and instantiating them
from configuration. This keeps the pipeline modular for future
detectors (weapons, car model, color).
"""

from typing import Callable, Dict

_REGISTRY: Dict[str, Callable] = {}


def register(name: str, factory: Callable):
    _REGISTRY[name] = factory


def get(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Detector plugin not registered: {name}")
    return _REGISTRY[name]


def available():
    return list(_REGISTRY.keys())

"""Name -> class registries for models, compressors, and datasets, so configs can refer to components by name."""
from collections import defaultdict

_REGISTRY: dict[str, dict[str, type]] = defaultdict(dict)


def register(kind: str, name: str):
    def decorate(cls):
        if name in _REGISTRY[kind]:
            raise KeyError(f"{kind} '{name}' is already registered")
        _REGISTRY[kind][name] = cls
        cls.registry_name = name
        return cls
    return decorate


def lookup(kind: str, name: str) -> type:
    if name not in _REGISTRY[kind]:
        raise KeyError(f"unknown {kind} '{name}'; known: {sorted(_REGISTRY[kind])}")
    return _REGISTRY[kind][name]


def build(kind: str, name: str, **kwargs):
    return lookup(kind, name)(**kwargs)


def names(kind: str) -> list[str]:
    return sorted(_REGISTRY[kind])

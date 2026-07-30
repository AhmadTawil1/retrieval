"""Embedding models behind one interface.

Two models are registered: "small" (384d) and "base" (768d), same architecture
family (BAAI/bge-*-en-v1.5) so the small/base contrast is dimension, not
architecture, per REFERENCE.md open decisions.

Revisions are pinned, but not by a hash typed into this file from a web search:
the exact commit SHA each model resolves to is recorded the first time it's
loaded, into configs/model_pins.yaml, and every load after that passes that
frozen revision explicitly. Delete a line from that file to intentionally
re-pin to whatever "main" currently resolves to.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from sentence_transformers import SentenceTransformer

PINS_PATH = Path("configs/model_pins.yaml")

MODEL_IDS = {
    "small": "BAAI/bge-small-en-v1.5",
    "base": "BAAI/bge-base-en-v1.5",
}

DIMENSIONS = {
    "small": 384,
    "base": 768,
}

_LOADED: dict[str, SentenceTransformer] = {}


def _load_pins() -> dict[str, str]:
    if PINS_PATH.exists():
        return yaml.safe_load(PINS_PATH.read_text()) or {}
    return {}


def _save_pin(name: str, revision: str) -> None:
    pins = _load_pins()
    pins[name] = revision
    PINS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PINS_PATH.write_text(yaml.safe_dump(pins, sort_keys=True))


def _resolve_revision(model_id: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(model_id).sha


def get_model(name: str) -> SentenceTransformer:
    """name is "small" or "base". Loads (and pins, on first call) the registered model."""
    if name in _LOADED:
        return _LOADED[name]
    if name not in MODEL_IDS:
        raise ValueError(f"unknown embedding model {name!r}, expected one of {list(MODEL_IDS)}")

    model_id = MODEL_IDS[name]
    pins = _load_pins()
    revision = pins.get(name)
    if revision is None:
        revision = _resolve_revision(model_id)
        _save_pin(name, revision)

    model = SentenceTransformer(model_id, revision=revision)
    _LOADED[name] = model
    return model


def get_revision(name: str) -> str:
    """The pinned revision for `name`, resolving and freezing it if not pinned yet."""
    pins = _load_pins()
    if name in pins:
        return pins[name]
    get_model(name)
    return _load_pins()[name]


def embed(texts: list[str], name: str) -> "list[list[float]]":
    model = get_model(name)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


if __name__ == "__main__":
    vecs = embed(["hello world", "retrieval configuration transfer"], "small")
    print(f"small -> {len(vecs)} vectors of dim {len(vecs[0])}, pinned at {get_revision('small')}")

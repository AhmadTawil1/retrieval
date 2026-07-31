"""Embedding models behind one interface.

Two models are registered: "small" (384d) and "base" (768d), same architecture
family (BAAI/bge-*-en-v1.5) so the small/base contrast is dimension, not
architecture, per REFERENCE.md open decisions.

Revision pinning lives in pins.py, which every model load in the project goes
through — embeddings here, reranker and generator in pipeline.py. It used to
live in this module and covered only these two models, which is how the A100
sweep ended up with three of its four models unrecorded (LOG.md, Day 4).
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

import pins

PINS_PATH = pins.PINS_PATH

MODEL_IDS = {
    "small": "BAAI/bge-small-en-v1.5",
    "base": "BAAI/bge-base-en-v1.5",
}

DIMENSIONS = {
    "small": 384,
    "base": 768,
}

_LOADED: dict[str, SentenceTransformer] = {}


def get_model(name: str) -> SentenceTransformer:
    """name is "small" or "base". Loads (and pins, on first call) the registered model."""
    if name in _LOADED:
        return _LOADED[name]
    if name not in MODEL_IDS:
        raise ValueError(f"unknown embedding model {name!r}, expected one of {list(MODEL_IDS)}")

    model_id = MODEL_IDS[name]
    model = SentenceTransformer(model_id, revision=pins.revision_for(model_id))
    _LOADED[name] = model
    return model


def get_revision(name: str) -> str:
    """The pinned revision for `name`, resolving and freezing it if not pinned yet."""
    return pins.revision_for(MODEL_IDS[name])


def embed(texts: list[str], name: str) -> "list[list[float]]":
    model = get_model(name)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


if __name__ == "__main__":
    vecs = embed(["hello world", "retrieval configuration transfer"], "small")
    print(f"small -> {len(vecs)} vectors of dim {len(vecs[0])}, pinned at {get_revision('small')}")

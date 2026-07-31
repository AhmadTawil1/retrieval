"""FAISS index plus a sidecar metadata table, so retrieval can return chunk text
and its source span (doc_id, char_start, char_end), not just a bare vector ID."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from .chunker import Chunk


@dataclass(frozen=True)
class Hit:
    doc_id: str
    char_start: int
    char_end: int
    text: str
    score: float


class Store:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.meta: list[Chunk] = []

    @classmethod
    def build(cls, chunks: list[Chunk], embeddings: list[list[float]]) -> "Store":
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        vecs = np.asarray(embeddings, dtype="float32")
        store = cls(dim=vecs.shape[1])
        store.index.add(vecs)
        store.meta = list(chunks)
        return store

    def search(self, query_vec: list[float], top_k: int) -> list[Hit]:
        q = np.asarray([query_vec], dtype="float32")
        scores, ids = self.index.search(q, top_k)
        hits = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            chunk = self.meta[idx]
            hits.append(Hit(chunk.doc_id, chunk.char_start, chunk.char_end, chunk.text, float(score)))
        return hits

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        meta = [
            {"doc_id": c.doc_id, "char_start": c.char_start, "char_end": c.char_end, "text": c.text}
            for c in self.meta
        ]
        (path / "meta.jsonl").write_text("\n".join(json.dumps(m) for m in meta))

    @classmethod
    def load(cls, path: Path) -> "Store":
        index = faiss.read_index(str(path / "index.faiss"))
        store = cls.__new__(cls)
        store.index = index
        store.meta = [
            Chunk(m["doc_id"], m["char_start"], m["char_end"], m["text"])
            for m in (json.loads(line) for line in (path / "meta.jsonl").read_text().splitlines())
        ]
        return store

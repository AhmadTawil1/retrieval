"""answer(query, cfg) -> {answer, retrieved_chunks}.

Retrieval (+ optional cross-encoder rerank) + generation, behind one config-driven
entry point. The generator is fixed and unswept on purpose (REFERENCE.md: this is
a retrieval-configuration study, not a generation-quality study) — a 3B-class
instruct model, greedy decoding, max_new_tokens=128, seed fixed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

import embed
from store import Hit, Store

GENERATOR_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
MAX_NEW_TOKENS = 128
SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TEMPLATE = (
    "Answer the question using ONLY the context below. If the context does not "
    "contain the answer, reply exactly: \"I don't know based on the provided context.\"\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

_generator_model = None
_generator_tokenizer = None
_reranker = None


@dataclass(frozen=True)
class Config:
    chunk_size: int
    overlap: float
    embed_model: str  # "small" | "base"
    top_k: int
    reranker: str  # "off" | "cross-encoder"


def config_id(cfg: Config) -> str:
    """Deterministic ID for a config, stable across processes and cards
    (must match across A100/L4 records per M01-RETRIEVAL.md §3)."""
    payload = json.dumps(dataclasses.asdict(cfg), sort_keys=True)
    return "sha1:" + hashlib.sha1(payload.encode()).hexdigest()[:12]


def _get_generator():
    global _generator_model, _generator_tokenizer
    if _generator_model is None:
        _generator_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_ID)
        _generator_model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL_ID, torch_dtype="auto").to(DEVICE)
    return _generator_model, _generator_tokenizer


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_ID)
    return _reranker


def embed_query(query: str, cfg: Config) -> list[float]:
    """Stage 1: embed."""
    return embed.embed([query], cfg.embed_model)[0]


def search(query_vec: list[float], store: Store, cfg: Config) -> list[Hit]:
    """Stage 2: search. Over-fetches when reranking so there's something to reorder."""
    fetch_k = cfg.top_k * 4 if cfg.reranker == "cross-encoder" else cfg.top_k
    return store.search(query_vec, top_k=fetch_k)


def rerank(query: str, hits: list[Hit], cfg: Config) -> list[Hit]:
    """Stage 3: rerank. A no-op pass-through when cfg.reranker == "off" — run_cell.py
    still times this stage, so an "off" cell's rerank cost should measure ~0."""
    if cfg.reranker != "cross-encoder" or not hits:
        return hits
    reranker = _get_reranker()
    scores = reranker.predict([(query, h.text) for h in hits])
    return [h for _, h in sorted(zip(scores, hits), key=lambda p: p[0], reverse=True)]


def retrieve(query: str, store: Store, cfg: Config) -> list[Hit]:
    q_vec = embed_query(query, cfg)
    hits = search(q_vec, store, cfg)
    hits = rerank(query, hits, cfg)
    return hits[: cfg.top_k]


def generate(query: str, hits: list[Hit]) -> tuple[str, int]:
    """Stage 4: generate. Returns (answer_text, n_new_tokens) — the token
    count is what run_cell.py needs for tok_per_s."""
    model, tokenizer = _get_generator()
    context = "\n\n".join(f"[{h.doc_id}] {h.text}" for h in hits)
    prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(DEVICE)

    torch.manual_seed(SEED)
    output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), len(new_tokens)


def answer(query: str, store: Store, cfg: Config) -> dict:
    hits = retrieve(query, store, cfg)
    text, _ = generate(query, hits)
    return {
        "answer": text,
        "retrieved_chunks": [
            {"doc_id": h.doc_id, "char_start": h.char_start, "char_end": h.char_end, "text": h.text, "score": h.score}
            for h in hits
        ],
    }


if __name__ == "__main__":
    import chunker

    cfg = Config(chunk_size=512, overlap=0.15, embed_model="small", top_k=5, reranker="off")
    chunks = chunker.chunk_corpus(Path("corpus"), size=cfg.chunk_size, overlap=cfg.overlap)
    vecs = embed.embed([c.text for c in chunks], cfg.embed_model)
    store = Store.build(chunks, vecs)

    for label, q in [
        ("in-corpus", "What is retrieval-augmented generation?"),
        ("out-of-corpus", "What is the capital of France?"),
    ]:
        result = answer(q, store, cfg)
        print(f"--- {label} ---")
        print("Q:", q)
        print("A:", result["answer"])
        print()

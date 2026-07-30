# Does the Pareto-optimal RAG configuration transfer across GPU tiers?

Draft. Sections are filled in on the day the thing they describe is built or measured
(REFERENCE.md §10.1) — a section with no content yet is marked `[pending: day N]`.

---

## Abstract

`[pending: day 7 — depends on the verdict]`

---

## 1. Introduction

Configuration-search papers for RAG pipelines publish a recommended configuration
benchmarked on the hardware the authors happened to have. Practitioners then copy
that recommendation onto whatever hardware they can afford. Nobody has
systematically tested whether the recommendation survives that move — if it
doesn't, every published "optimal configuration" carries an unstated hardware
precondition. This measurement asks: **does the Pareto-optimal RAG configuration
transfer across GPU tiers, or does the ranking of configurations invert?**

**Hypothesis 1.** The Pareto-optimal RAG configuration is not hardware-invariant.
Chunk size and top-k keep their optimal settings across GPU tiers, but
cross-encoder reranking flips from Pareto-optimal on the A100 to
Pareto-dominated on the L4 — its latency cost grows faster than the recall it
buys.

**Refuted if** both frontiers contain the same configurations in the same
order — identical composition means the recommendation transfers, and the
thesis fails in this setting.

Contribution bullets: `[pending: day 7 — depends on knowing what the contributions turned out to be]`

---

## 2. Related work

Multi-objective RAG configuration search is a crowded literature; this work
does not claim to be the first to locate a Pareto frontier over retrieval
configurations. **syftr** performs multi-objective search across agentic and
non-agentic RAG configurations, balancing latency, accuracy and cost against
each other on a single hardware target. **RAG-Stack** identifies the
quality–performance Pareto frontier by navigating the joint
algorithm–system design space, and is the closest prior work to acknowledge
that hardware options exist at all — but it explores primarily from the
algorithm side, treating the system as a fixed backdrop rather than a swept
variable. **RAGSmith** runs evolutionary search over complete pipelines
rather than greedy per-module selection, and separately reports that its
recommended pipeline is sensitive to the dataset it was tuned on — a
sensitivity result adjacent to, but distinct from, hardware sensitivity.
**METIS** performs joint algorithm/system exploration for retrieval-augmented
pipelines, again on fixed hardware. Alongside these, a growing line of
cost–latency–quality benchmark papers treats the three-way tradeoff itself as
the object of study, without varying the hardware the tradeoff is measured on.

Across all five, the hardware tier is a constant, not a knob. RAG-Stack comes
closest to raising the question, since it is aware that system-side choices
exist, but even there the comparison is across algorithms on one machine, not
across machines running the same algorithm. None of this work asks whether
the configuration it recommends would still be recommended on a smaller GPU.

**Existing work locates the configuration frontier on fixed hardware; whether
the frontier's composition is stable across hardware tiers has not been
tested.**

---

## 3. Method

### 3.1 System under test

**Corpus.** 15 topic PDFs on practical GenAI/LLM-engineering subjects
(agentic AI, APIs, Docker, FastAPI, fine-tuning, GenAI systems, Hugging Face,
LangGraph, LlamaIndex, LLMs, multimodal models, PyTorch, RAG, uv, vLLM) — one
subject, deliberately, since a single-topic corpus retrieves harder and more
realistically than 15 unrelated documents. 72 pages, 35,751 tokens total
(counted with the `base` embedding model's tokenizer). Extracted with `pypdf`
6.14.2. `corpus_sha` (SHA1 over sorted filename+bytes pairs):
`2b3486e1671811722b6f256f04257667eef78ccd`.

**Chunking (`chunker.py`).** Recursive character splitting, parameterised by
`(size, overlap)` where `size` is in tokens. Separators are tried
coarse-to-fine — paragraph break, line break, sentence, word, raw character —
descending into a piece only when it is still over budget; overlap is
produced by carrying trailing atoms from one chunk into the next rather than
re-slicing. Every chunk keeps `{doc_id, char_start, char_end}` relative to
that document's full extracted text, which is what lets gold labels (day 2)
be defined as source-document spans instead of chunk IDs — chunk IDs would
not survive a change in chunk size, and three of the five grid knobs change
the chunking.

**Embeddings (`embed.py`).** `sentence-transformers` behind one interface,
two models registered:

| Name | Model | Dim | Revision |
|---|---|---|---|
| `small` | `BAAI/bge-small-en-v1.5` | 384 | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| `base` | `BAAI/bge-base-en-v1.5` | 768 | `[pending: resolves on first use of the base model, see configs/model_pins.yaml]` |

Same model family for both, so the small/base contrast is dimension, not
architecture. Revisions are not hand-typed: the exact commit SHA each model
resolves to is captured the first time it is loaded and frozen into
`configs/model_pins.yaml`; every later load passes that frozen revision
explicitly, so a change upstream on the Hub cannot silently change what "the
embedding model" means mid-sweep.

**Vector store (`store.py`).** FAISS `IndexFlatIP` over L2-normalised
embeddings (inner product = cosine similarity), with a JSONL sidecar metadata
table keyed to the same row order as the index. Retrieval returns chunk text
and its source span, not a bare vector ID.

**Reranker.** `cross-encoder/ms-marco-MiniLM-L6-v2` — the lighter of the two
candidates under consideration, chosen deliberately because it makes
Hypothesis 1 harder to confirm.

**Generation (`pipeline.py`).** `Qwen/Qwen2.5-3B-Instruct`, fixed and unswept
across the entire grid: this is a retrieval-configuration study, not a
generation-quality study, and a larger generator would cost sweep days
(≈8,600 generations across the full grid) without bearing on the hypothesis.
Answer-only-from-context prompt:

```
Answer the question using ONLY the context below. If the context does not
contain the answer, reply exactly: "I don't know based on the provided context."

Context:
{context}

Question: {question}

Answer:
```

Greedy decoding (`do_sample=False`, `num_beams=1`), `max_new_tokens=128`,
seed fixed at `1337`.

### 3.2 The configuration grid

The first three knobs determine the index; the last two are query-time only —
which is what makes 96 cells cost only 12 index builds.

| Knob | Levels | Count | Affects |
|---|---|---|---|
| Chunk size | 256 / 512 / 1024 tokens | 3 | Index — rebuild required |
| Chunk overlap | 0 / 15% | 2 | Index — rebuild required |
| Embedding model | small (384d) / base (768d) | 2 | Index — rebuild required |
| Top-k | 3 / 5 / 10 / 20 | 4 | Query time only |
| Reranker | off / cross-encoder | 2 | Query time only — the hypothesis |

Indices to build: 3 × 2 × 2 = **12**. Cells per card: 12 × 4 × 2 = **96**.
Total runs across both cards: 192. The full grid, with a deterministic
`config_id` per cell (SHA1 over the sorted config, so it matches across
cards), is frozen in `configs/grid.yaml` as of today and is not touched again
except under the cut order in `M01-RETRIEVAL.md` §2, decided before a sweep
starts rather than mid-sweep.

### 3.3 Gold set construction

`[pending: day 2]`

### 3.4 Harness and measurement methodology

`[pending: day 3]`

### 3.5 Hardware and software

`[pending: day 3, versions filled days 4–5]`

### 3.6 Metric definitions

`[pending: day 2]`

### 3.7 Reproduction

`[pending: day 3]`

### 3.8 Handling of unrunnable cells

`[pending: day 5]`

---

## 4. Results

`[pending: day 6]`

---

## 5. Threats to validity

`[pending: days 2, 5, 7]`

---

## 6. Conclusion

`[pending: day 7]`

---

## References

`[pending: day 7]`

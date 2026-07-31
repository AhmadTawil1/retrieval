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

30 questions, hand-labelled by a single annotator (Ahmad Tawil) reading the
corpus directly — not generated. Gold labels are **character spans in the
source document**, `{doc_id, char_start, char_end}`, never chunk IDs: chunk
IDs only exist relative to one chunk size, and three of the five grid knobs
change the chunking, so a label made at 512 tokens would be meaningless at
256 or 1024 (§4.1). Spans are measured against the same `pypdf`-extracted
text the pipeline actually indexes, not the PDF's visual layout — the two
can differ (missing spaces, mangled special characters) — using
`label_helper.py`, which searches a document's extracted text for a phrase
and returns its exact `char_start`/`char_end` plus surrounding context, so
offsets are never hand-counted. A retrieved chunk counts as relevant to a
gold span if it overlaps that span by ≥ 50% of the span's length (§3.6);
that threshold is a defensible choice only because it is stated plainly
here, not tuned after seeing results.

Questions were allocated by type before any question was written
(`data/GOLD-PLAN.md`): 7 single-sentence-answer, 9 paragraph-answer, 8
lexically-confusable-across-documents (phrased to avoid the source
document's own terminology, so a near-lexical match against the wrong
document is a live risk), and 6 requiring two documents — 30 questions, 38
gold spans (multi-document questions carry 2–3 spans each), spanning all 15
corpus documents. Span length: 38–414 characters, median 193.

Two annotation passes. The first review pass found two mislabelled
confusable-document pairs (corrected) and, more consequentially, five
questions with high content-word overlap against their own gold span (47–62%
— one question reused six of the span's terms in the span's order). **A
question phrased in its answer span's own vocabulary is retrievable by
near-lexical match regardless of configuration** — recall saturates across
the entire grid, both cards produce identical frontiers, and that reads as a
transfer result (Hypothesis 1 refuted) when it is really a saturated
instrument that never gave any configuration a chance to fail. All five were
rewritten in the second pass to describe the answer without its source
terminology, without changing which span they target.

Span offsets were located against the same `pypdf`-extracted text the
pipeline indexes, not against a cleaned reading of the PDF — 13 of the
initially-quoted spans didn't resolve on the first pass because of
extraction artifacts (code listings with line numbers interleaved into the
text, identifiers extracted letter-spaced, punctuation that swallows
adjacent spaces). Every span was re-resolved and verified against the actual
extracted text before freezing. `gold_sha = 3afa042d8d9cd6784e8e1b049e4f4f6c1d45709c`.

### 3.4 Harness and measurement methodology

One config in, one JSON-line record out (`run_cell.py`), matching a fixed
schema exactly so a card's results are never hand-assembled from a
spreadsheet. Four stages are timed individually — embed, search, rerank,
generate — each bracketed by `torch.cuda.synchronize()` immediately before
the stop timestamp; GPU kernels launch asynchronously, so an unsynchronised
stop timestamp measures how fast the CPU could issue instructions, not how
long the GPU took. `rerank` is timed even on `reranker=off` cells, where it
is a no-op pass-through — its cost should measure ~0, which is itself a
check that the instrumentation is honest.

**Warmup and repeats.** The first 3 gold queries run once each through the
full stage pipeline before any timing starts, discarded — this absorbs
lazy model loading and any one-time kernel/JIT warmup, none of which is the
number a config's latency should be judged on. Every one of the 30 gold
queries is then run 3 times end to end; p50 and p95 are computed over all 90
per-query-repeat measurements for a cell, not over 30 per-query means — a
mean-of-means would hide exactly the tail behavior p95 exists to catch.

**p50/p95, never the mean.** Rented hosts have noisy neighbours and
asymmetric latency tails; a mean is pulled around by outliers in a way that
misrepresents what most queries actually experience (REFERENCE.md ground
rule 4).

**VRAM.** `torch.cuda.reset_peak_memory_stats()` at the start of a cell,
`torch.cuda.max_memory_allocated()` at the end — the high-water mark across
the whole cell, not a single snapshot.

**OOM handling.** A cell that raises a CUDA out-of-memory error is caught,
VRAM is cleared, and the record is written with `run.status="oom"` and no
`quality`/`cost` block, rather than crashing the sweep or being silently
skipped — a refusal is a finding (M01-RETRIEVAL.md §3), and `validate_record`
in `run_cell.py` treats a well-formed OOM record as valid on its own terms.

**Index cache.** Keyed on `(chunk_size, overlap, embed_model)` — the three
knobs that actually change the index — so the sweep orders by index and
rebuilds 12 times, not 96 (§4.2).

**Provenance.** Every record's `prov` block (`provenance.py`) is stamped
fresh at run time: GPU name and driver (via `nvidia-smi`, `"none (CPU)"` off
GPU), CUDA/torch/sentence-transformers/faiss versions, git SHA, and the
frozen `corpus_sha`/`gold_sha` — never typed from memory, never reused from
a previous run.

### 3.5 Hardware and software

| Tier | Card | Arch | SM | VRAM | Mem bandwidth |
|---|---|---|---|---|---|
| Strong | A100 80GB | Ampere | SM80 | 80 GB | ~2,039 GB/s |
| Weak | L4 24GB | Ada | SM89 | 24 GB | ~300 GB/s |

Bandwidth ratio ≈ 6.8× (REFERENCE.md §3) — deliberately a wide spread, so a
configuration crossing is visible rather than lost in noise. Driver, CUDA,
and library versions are `[pending: days 4–5 — filled from each run's actual
`prov` stamp once rented, never typed from memory]`.

### 3.6 Metric definitions

All four metrics are computed from spans, via one relevance rule
(`relevance.py`): retrieved chunk `c` is relevant to gold span `g` iff
`overlap(c, g) / len(g) ≥ 0.5`, and irrelevant by definition if `c` and `g`
are in different documents.

- **recall@k** — over a query with gold spans `G`, 1 if any of the top-k
  retrieved chunks is relevant to any span in `G`, else 0; averaged over all
  30 queries. Reported at k=5 (headline) and k=10 (distinguishes bad ranking
  from bad retrieval — a config can fail at 5 and recover at 10).
- **MRR** — reciprocal rank of the first retrieved chunk relevant to any
  span in `G` (0 if none); averaged over all 30 queries. Sensitive to
  ordering, which is exactly what a reranker changes and recall@k is blind
  to.
- **Context precision** — fraction of the top-5 retrieved chunks relevant to
  any span in `G`; averaged over all 30 queries. Falls as top-k rises; this
  is the cost of a large k that recall alone hides.

Implementation: `eval.py`, pure functions with no pipeline dependency,
unit-tested against hand-built boundary cases (`tests/test_eval.py`) —
including the recall@k off-by-one at exactly rank k, and the is_relevant
boundary at exactly the 0.5 threshold.

### 3.7 Reproduction

Any single cell regenerates with one command:

```
uv run python run_cell.py <config_id>
```

`config_id` is the SHA1-derived ID frozen in `configs/grid.yaml` (§3.2); the
command builds or reuses the cached index for that cell's `(chunk_size,
overlap, embed_model)`, runs the full 30-query/3-repeat protocol, and prints
one JSON record matching §3's schema, provenance included. A full card's
sweep is the same command over all 96 `config_id`s in `configs/grid.yaml`,
skipping any already present in that card's result file (day 4).

### 3.8 Handling of unrunnable cells

`[pending: day 5]`

---

## 4. Results

`[pending: day 6]`

---

## 5. Threats to validity

**Single annotator, no inter-annotator agreement figure.** All 30 gold
questions and spans were labelled by one person (Ahmad Tawil), with no
second annotator to measure agreement against. Ambiguous cases were
resolved by that same person's judgment call, not adjudicated.

**Question distribution chosen by the same person who built the corpus.**
The 15-document corpus and the 30 gold questions were selected by the same
individual, who therefore already knew what the corpus could answer before
writing questions against it — a risk of unconsciously favoring questions
the retrieval system is likely to handle well.

**Lexical-overlap saturation was a live risk in the gold set, mitigated but
not eliminated.** A question phrased in its own answer span's vocabulary is
retrievable by near-lexical match at any configuration, which would collapse
both cards' frontiers to the same shape for a reason unrelated to hardware
(§3.3). Five such questions were caught and rewritten during review; the
review process itself (one annotator, checking word overlap against a
threshold chosen after noticing the problem) has no independent check behind
it, so residual saturation in some of the 30 cannot be fully ruled out.

`[pending: days 5, 7 — remaining entries: two cards is not a continuum, single seed, and whatever day 6 exposes]`

---

## 6. Conclusion

`[pending: day 7]`

---

## References

`[pending: day 7]`

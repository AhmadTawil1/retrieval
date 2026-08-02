# Does the Pareto-optimal RAG configuration transfer across GPU tiers?

Draft. Sections are filled in on the day the thing they describe is built or measured
(REFERENCE.md §10.1) — a section with no content yet is marked `[pending: day N]`.

---

## Abstract

Configuration-search papers for RAG pipelines report a single Pareto-optimal
recommendation, benchmarked on whichever GPU the authors had, and say
nothing about whether that recommendation survives a move to cheaper
hardware. We measure it directly: an identical 96-configuration grid
(chunk size, overlap, embedding model, top-k, reranker) swept against one
corpus and one hand-labelled gold set on an A100-SXM4-40GB and an L4,
provenance-stamped per record. The result is partial, not the clean
confirmation or refutation pre-registered. Pareto-frontier membership does
change between cards — 4 members on the A100, 6 on the L4, 3 shared — but
the knob that moves is the embedding model, not the cross-encoder reranker
the pre-registered hypothesis named. The reranker's own latency does
degrade disproportionately on the weaker card (1.40× versus 1.04–1.17× for
every other pipeline stage) — the predicted economics are real and
measurable — but at only ~1–1.5% of a generation-dominated cell's latency,
that disproportion is not yet large enough to flip frontier membership.
Kendall tau between the two cards' latency rankings is 0.80: most of the
ordering transfers, not all of it.

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

**Contributions.**

- A reproducible, self-describing measurement harness for comparing RAG
  configurations across GPU tiers: every record carries its own git SHA,
  corpus/gold hashes, and model revisions, so a result never depends on
  external state to be trusted (§3.4). Two provenance gaps were found and
  closed *during* this measurement (LOG.md Days 4–5), which is itself
  evidence the discipline is load-bearing, not decorative.
- Evidence that Pareto-frontier membership is not hardware-invariant, but
  that the specific mechanism proposed in advance can miss even when the
  broader claim holds: the reranker, the knob expected to be the risk, did
  not change status; a quieter knob (embedding-model size) did. Hardware
  sensitivity is not confined to the component that looks computationally
  expensive.
- A per-stage attribution showing the reranker's predicted cost/benefit
  imbalance *is* measurable (a disproportionate 1.40× latency scaling
  versus 1.04–1.17× elsewhere) even in a setting where it is not yet
  decisive for frontier membership — because generation dominates per-cell
  latency here. This names the condition (a smaller generator, or a
  heavier reranker) under which the originally-predicted mechanism would be
  expected to become frontier-decisive, rather than leaving "not confirmed"
  as a dead end.

---

## 2. Related work

Multi-objective RAG configuration search is a crowded literature; this work
does not claim to be the first to locate a Pareto frontier over retrieval
configurations. **syftr** (Conway et al., 2025) performs multi-objective
Bayesian optimization across agentic and non-agentic RAG flows, balancing
accuracy and cost against each other on a single hardware target.
**RAG-Stack** (Jiang, 2025) identifies the quality–performance
Pareto frontier by navigating the joint algorithm–system design space, and
is the closest prior work to acknowledge that hardware options exist at
all — but it explores primarily from the algorithm side, treating the
system as a fixed backdrop rather than a swept variable. **RAGSmith**
(Kartal et al., 2025) runs evolutionary search over complete pipelines
rather than greedy per-module selection, and separately reports that its
recommended pipeline is sensitive to the dataset it was tuned on — a
sensitivity result adjacent to, but distinct from, hardware sensitivity.
**METIS** (Ray et al., 2025) performs joint algorithm/system exploration for
retrieval-augmented pipelines, again on fixed hardware. Alongside these, a
growing line of cost–latency–quality benchmark papers treats the three-way
tradeoff itself as the object of study, without varying the hardware the
tradeoff is measured on. The underlying retrieval-augmented generation
approach itself is Lewis et al. (2020).

Across all five, the hardware tier is a constant, not a knob. RAG-Stack comes
closest to raising the question, since it is aware that system-side choices
exist, but even there the comparison is across algorithms on one machine, not
across machines running the same algorithm. None of this work asks whether
the configuration it recommends would still be recommended on a smaller GPU.

**Existing work locates the configuration frontier on fixed hardware; whether
the frontier's composition is stable across hardware tiers has not been
tested.** The finding here is the same *shape* as an earlier result of ours
(Tawil, 2026): that paper found the optimal RAG *strategy* (agentic versus
naive) depends on model scale; this one finds the optimal RAG
*configuration* depends on hardware tier. Both say the same thing about
what "optimal" means in this literature — it is conditional on something
the recommending paper held fixed.

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

**Embeddings (`embed.py`).** `sentence-transformers` (Reimers & Gurevych,
2019) behind one interface, two BGE models (Xiao et al., 2023) registered:

| Name | Model | Dim | Revision |
|---|---|---|---|
| `small` | `BAAI/bge-small-en-v1.5` | 384 | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` (measured) |
| `base` | `BAAI/bge-base-en-v1.5` | 768 | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` (inferred†) |

Same model family for both, so the small/base contrast is dimension, not
architecture. Revisions are not hand-typed: the exact commit SHA each model
resolves to is captured the first time it is loaded and frozen into
`configs/model_pins.yaml`; every later load passes that frozen revision
explicitly, so a change upstream on the Hub cannot silently change what "the
embedding model" means mid-sweep.

**† Measured vs. inferred (LOG.md, Day 4).** `small`'s revision is a
measurement: `configs/model_pins.yaml` is git-tracked, and the commit it was
frozen at (`04caf49`) is the `git_sha` recorded on all 96 A100 records — so
the sweep provably loaded that exact revision. `base`'s revision, and the
reranker's and generator's below, are not: the pin file is written at
runtime on whichever machine loads a model first, and that machine was a
Colab VM deleted before its copy of the file could be read back. These three
were recovered by resolving `main` on 1 Aug 2026 and are believed equal to
what the sweep actually loaded because each source repo's last update
predates the 31 Jul 2026 sweep by 11 months to 2.4 years — assuming no
force-push or reverted commit on the Hub in between (§5). From the L4 sweep
onward this gap is closed: `provenance.stamp()` writes every model's
resolved revision into `prov.model_revisions` on every record, so a pin can
no longer be lost with the machine that wrote it.

**Vector store (`store.py`).** FAISS (Douze et al., 2024) `IndexFlatIP` over L2-normalised
embeddings (inner product = cosine similarity), with a JSONL sidecar metadata
table keyed to the same row order as the index. Retrieval returns chunk text
and its source span, not a bare vector ID.

**Reranker.** `cross-encoder/ms-marco-MiniLM-L6-v2` — a Sentence-BERT
cross-encoder (Reimers & Gurevych, 2019) trained on MS MARCO (Bajaj et al.,
2016) — the lighter of the two candidates under consideration, chosen
deliberately because it makes Hypothesis 1 harder to confirm. Revision
`c5ee24cb16019beea0893ab7796b1df96625c6b8` (inferred†).

**Generation (`pipeline.py`).** `Qwen/Qwen2.5-3B-Instruct` (Qwen Team,
2024), fixed and unswept
across the entire grid: this is a retrieval-configuration study, not a
generation-quality study, and a larger generator would cost sweep days
(≈8,600 generations across the full grid) without bearing on the hypothesis.
Revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` (inferred†).
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
| Strong | **A100-SXM4-40GB** | Ampere | SM80 | 40 GB | ~1,555 GB/s |
| Weak | L4 24GB | Ada | SM89 | 24 GB | ~300 GB/s |

**Correction to REFERENCE.md's planning assumption.** The plan assumed an
A100 80GB card (~2,039 GB/s); the A100 actually provisioned (Colab Pro,
confirmed via `torch.cuda.get_device_name(0)` and cross-checked against
`nvidia-smi` in `provenance.py`'s stamp) is the 40GB SXM4 SKU, ~1,555 GB/s.
Bandwidth ratio against the L4 is therefore ≈5.2×, not the ≈6.8× originally
planned — still a wide spread, wide enough for a configuration crossing to
be visible, but the number reported here is the one actually measured, not
the one assumed at planning time. VRAM capacity (40GB) is not a constraint
for this study's workload regardless of the correction.

**Software, from each sweep's own `prov` stamp, not typed from memory.**

| | A100 | L4 |
|---|---|---|
| Driver | `580.82.07` | `580.82.07` |
| CUDA | `12.8` | `13.0` |
| `torch` | `2.11.0+cu128` | `2.13.0+cu130` |
| `sentence-transformers` | `5.6.0` | `5.6.1` |
| `faiss` | `1.14.3` | `1.14.3` |
| `git_sha` | `04caf49` | `3761de1` |

Model revisions (§3.1) are identical on both records — `small`
`5c38ec7c…`, `base` `a5beb1e3…`, reranker `c5ee24cb…`, generator
`aa8e7253…` — so despite the differing `git_sha`, both cards loaded exactly
the same four model weights.

**The `git_sha` gap is provenance-capture only, not a measured-path
change.** The pinning fix (`pins.py`, LOG.md Day 4) landed between the two
sweeps, so the L4 sweep ran at a later commit than the A100's. Checked
directly rather than assumed: `diff <(git show 04caf49:chunker.py)
<(git show 3761de1:retrieval/chunker.py)` and the same for `store.py`,
`eval.py`, `relevance.py` — every file the retrieval/generation path
actually executes — are byte-identical except two import-statement lines
(`from chunker import Chunk` → `from .chunker import Chunk`, and the
equivalent for `relevance`), a consequence of the Day-4→5 package
restructure, not a behavior change. §3.7 and §4 report both SHAs
explicitly rather than picking one.

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
uv run python scripts/run_cell.py <config_id>
```

`config_id` is the SHA1-derived ID frozen in `configs/grid.yaml` (§3.2); the
command builds or reuses the cached index for that cell's `(chunk_size,
overlap, embed_model)`, runs the full 30-query/3-repeat protocol, and prints
one JSON record matching §3's schema, provenance included. A full card's
sweep is the same command over all 96 `config_id`s in `configs/grid.yaml`,
skipping any already present in that card's result file (day 4).

### 3.8 Handling of unrunnable cells

Decided before the L4 sweep runs, while it is still unknown which cells (if
any) this affects — not after seeing whether the decision helps or hurts
Hypothesis 1.

A cell that raises a CUDA OOM is recorded with `status:"oom"` and its config,
cost, and quality fields absent (§3.4) — never retried at a reduced setting,
never silently dropped.

- **Excluded from that card's Pareto frontier.** An OOM'd cell has no
  p95/recall pair to plot, so it cannot be a frontier member by construction,
  not by a judgment call made after seeing which cells failed.
- **Not excluded from the paper.** Every OOM'd cell appears verbatim, config
  and error, in a refusals table (`results/refusals.md`, generated by
  `scripts/verify_pair.py --refusals`, never hand-typed), which becomes a
  table in §4. A card refusing a configuration outright — not just running it
  slower — is itself part of the hardware-dependence result (REFERENCE.md:
  "record every refusal").
- **A config that is Pareto-optimal on the A100 and OOMs on the L4 is not a
  missing data point to explain away.** It is the strongest available form of
  Hypothesis 1: the configuration does not merely lose the frontier on the
  weaker card, it cannot run there at all.
- **No substitution.** An OOM'd cell's absence is never filled by a
  different top-k/chunk-size/reranker setting standing in for it, and never
  by extrapolating from neighbouring cells.

---

## 4. Results

**Verdict: partial.** Frontier membership changed between A100 and L4 — the
two frontiers are not identical, so Hypothesis 1's refutation condition
(SCOPE.md) does not hold. But the knob that moved is `embed_model`, not the
reranker, and cross-encoder reranking appears on *both* cards' frontiers.
The specific mechanism Hypothesis 1 named — reranker flips from
Pareto-optimal to Pareto-dominated, chunk size and top-k unmoved — did not
happen. Full numbers below; all reproducible with `uv run python
scripts/analysis.py data/a100.jsonl data/l4.jsonl`.

**Figure 1** (`figs/fig1.pdf`) plots recall@5 against p95 latency for all 96
configs on both cards. Filled, ringed points are that card's Pareto
frontier; thin gray lines connect the same `config_id` across cards for
every frontier-relevant point, so a reader can trace one configuration's
move.

**Frontier membership table.** 7 configs sit on either frontier (89 sit on
neither, dominated on both cards):

| config_id | chunk | overlap | embed | top_k | reranker | recall@5 | A100 p95 | L4 p95 | delta |
|---|---|---|---|---|---|---|---|---|---|
| `9df54f160c03` | 256 | 0.0 | base | 3 | off | 0.833 | 5249ms | 5659ms | both |
| `71d5acecfa94` | 256 | 0.0 | base | 5 | off | 0.900 | 5329ms | 5672ms | both |
| `7854342500f9` | 1024 | 0.0 | base | 5 | off | 0.967 | 5555ms | 6048ms | both |
| `93ce410a3c36` | 1024 | 0.15 | **small** | 3 | cross-encoder | 0.933 | 5474ms | 5876ms | **A100 only** |
| `0ca6aa10c184` | 1024 | 0.15 | **base** | 3 | cross-encoder | 0.933 | 5483ms | 5832ms | **L4 only** |
| `8edd4990fff0` | 256 | 0.0 | small | 3 | off | 0.667 | 5280ms | 5621ms | L4 only |
| `1da8ed40f4d1` | 256 | 0.0 | small | 3 | cross-encoder | 0.700 | 5298ms | 5623ms | L4 only |

A100 frontier: 4 members. L4 frontier: 6 members. 3 shared.

**Refusal table.** Empty on both cards — zero OOMs on either A100 or L4
(`results/refusals.md`), contrary to the plan's expectation at
chunk=1024×top_k=20×cross-encoder×base (LOG.md Day 5).

**Knob-by-knob.** For every knob, the *set* of values appearing among
frontier configs is identical between cards (both cards' frontiers use
chunk_size {256, 1024}, overlap {0.0, 0.15}, embed_model {small, base},
top_k {3, 5}, and reranker {off, cross-encoder}) — at that coarse level,
nothing moved. What actually moved is visible only by pairing specific
configs: `93ce410a3c36` (embed=small) and `0ca6aa10c184` (embed=base) are
otherwise-identical (chunk=1024, overlap=0.15, top_k=3, reranker=on,
recall=0.933 on both cards) — the only difference between them is
`embed_model`, and which one wins the frontier slot swaps between cards.
`embed_model` is the knob that moved, not the reranker.

**This swap is small in absolute terms and is flagged, not oversold.** The
latency gap deciding it is 9ms (A100: small faster by 5474 vs 5483) and
44ms (L4: base faster by 5832 vs 5876) — under 1% of the ~5,500–5,900ms
total. With p95 over 90 measurements and no repeated-sweep variance
estimate, this is within the range a rented host's noisy neighbours could
produce on a re-run (§5). The frontier-membership *result* is real and
reproducible from the recorded data; whether `embed_model` would swap the
same way on a second L4 sweep is not established here.

**Kendall tau** (Kendall, 1938) **(latency ordering, A100 vs L4, n=96): 0.802.** Strong but not
perfect — most of the grid's speed ranking transfers, with enough
reordering to produce the frontier delta above.

**Per-stage attribution — the reranker's mechanism is present, just not
decisive here.** Reranking adds a mean +0.018 recall@5 (identical on both
cards, as expected — quality doesn't depend on hardware) at a real latency
cost. That cost does *not* scale like the rest of the pipeline: the
rerank stage's own p50 grows 1.40× from A100 to L4, against 1.14× for
generate, 1.04× for embed, and ~1.00× for search (FAISS search stays
CPU-bound regardless of GPU) — an overall p95 ratio of 1.14× across shared
configs. The reranker is the one stage whose cost degrades
disproportionately on the weaker card, which is exactly Hypothesis 1's
claimed economics. It does not flip any config's frontier status here
because generation dominates total latency (rerank is ~1–1.5% of a cell's
p95; the disproportionate slowdown has nowhere large enough to act). A
generator with a smaller share of per-cell latency, or a heavier reranker,
is where this mechanism would be expected to become frontier-decisive.

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

**Three of four A100 model revisions are inferred, not measured.** Revision
pinning originally covered only the two embedding models, and only wrote the
pin to a file on the machine that loaded it first — that machine was a Colab
VM, deleted after the sweep. `small`'s revision survives because
`configs/model_pins.yaml` is git-tracked and was committed before the sweep
ran; `base`, the reranker, and the generator's revisions do not, and were
reconstructed by resolving `main` on 1 Aug 2026 under the assumption that
each model's Hub repo was unchanged since its last update (11 months to 2.4
years before the sweep) and has not been force-pushed or had a commit
reverted in between (§3.1). The reranker is one of the two arms of
Hypothesis 1, so if this assumption is wrong for
`cross-encoder/ms-marco-MiniLM-L6-v2` specifically, the A100 measurement
reflects a different reranker than documented. The fix (`pins.py`,
LOG.md Day 4) makes every subsequent record self-describing — this threat
applies to the A100 card only, not to L4.

**Two cards is not a continuum.** The hardware axis has exactly two points,
A100 and L4 (§3.5) — this measurement can show whether the frontier's
composition changed between these two specific tiers, not how it changes
with GPU class in general. A monotonic story (the frontier shifts gradually
as hardware weakens) and a threshold story (nothing changes until some tier
between these two, then it flips) would produce identical results here.
Whatever Measurement 01 finds is a statement about A100-vs-L4, not about
hardware tiers as a continuum.

**Single seed.** `pipeline.generate`'s `torch.manual_seed(1337)` (§3.1) is
fixed once, not varied across repeats or between cards. This is a narrower
limitation than it first looks: decoding is greedy (`do_sample=False`), so
generation is already deterministic given the model weights and the seed —
the three repeats per cell measure timing variance, not generation variance,
and retrieval is seed-independent entirely (the seed is only read inside
`generate()`). The residual risk is narrow but real: a different seed is not
expected to change which chunks are retrieved or the shape of the recall/p95
frontier, only the exact wording of refused-vs-answered generations at the
margin, which this measurement does not score.

**One corpus.** All 192 measurements (96 cells × 2 cards) run against a
single 15-document, single-subject corpus (§3.1). Whether frontier
composition is corpus-dependent — a technical corpus with code listings and
structured documents, versus prose-heavy general text — is untested here; a
different corpus could change which configs are close enough to tie for a
knob like `embed_model` to flip the way it did (§4).

**One embedding family.** `small` and `base` are both `BAAI/bge-*-en-v1.5`
— same architecture and training recipe, differing only in dimension
(§3.1). The measured hardware-sensitivity is therefore about embedding
*size* within one family, not embedding *architecture*; whether a
different family (different tokenizer, pooling, or training objective)
would show the same or a different swap is not addressed by this
measurement.

**Rented hosts have noisy neighbours, and there is no repeated-sweep
variance estimate.** p50/p95 exist because of this (§3.4), but each of the
192 cells ran once per card, not N sweeps averaged — there is no measured
run-to-run variance to compare against. The `embed_model` frontier swap
(§4) turns on latency gaps of 9–44ms against a ~5,500–5,900ms cell — under
1%, exactly the scale host noise could produce. The result stands as
measured and is reproducible from the recorded data; whether it replicates
on a second L4 sweep is not established.

---

## 6. Conclusion

Do not assume a Pareto-optimal RAG configuration transfers unchanged from a
benchmarking GPU to a cheaper deployment card — even when it mostly does.
Here, the knob a practitioner would watch first, the reranker, because it
has an obvious latency cost, did not change status; a quieter knob
(which embedding-model size wins the margin) did, and only because two
configurations happened to be close enough to tie. The lesson is not
"watch the reranker" — it is that *any* near-tied pair on your frontier is
a hardware-transfer risk, not just the expensive-looking knob. Concretely:
before shipping, re-run your top handful of frontier candidates, not just
the single winner, on your actual deployment hardware — a config that
loses by a hair on the benchmarking card can win on the target one. And if
your generator is small enough that non-generation stages are a meaningful
share of latency, budget extra scrutiny for the reranker specifically: its
cost is shown here to degrade faster than the rest of the pipeline, even
though it was not decisive in this generator-dominated setup.

---

## References

Identifiers verified against primary sources (arXiv abstract pages fetched
directly) on 2 Aug 2026, not copied from a to-locate list.

1. Bajaj, P., Campos, D., Craswell, N., Deng, L., Gao, J., Liu, X.,
   Majumder, R., McNamara, A., Mitra, B., Nguyen, T., Rosenberg, M., Song,
   X., Stoica, A., Tiwary, S., & Wang, T. (2016). *MS MARCO: A Human
   Generated MAchine Reading COmprehension Dataset*. arXiv:1611.09268.
2. Conway, A., Dey, D., Hackmann, S., Hausknecht, M., Schmidt, M.,
   Steadman, M., & Volynets, N. (2025). *syftr: Pareto-Optimal Generative
   AI*. arXiv:2505.20266.
3. Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré,
   P.-E., Lomeli, M., Hosseini, L., & Jégou, H. (2024). *The Faiss
   library*. arXiv:2401.08281.
4. Jiang, W. (2025). *RAG-Stack: Co-Optimizing RAG Quality and Performance
   From the Vector Database Perspective*. arXiv:2510.20296.
5. Kartal, M. Y., Kose, S. K., Sevinç, K., & Aktas, B. (2025). *RAGSmith: A
   Framework for Finding the Optimal Composition of Retrieval-Augmented
   Generation Methods Across Datasets*. arXiv:2511.01386.
6. Kendall, M. G. (1938). A New Measure of Rank Correlation. *Biometrika*,
   30(1–2), 81–93. https://doi.org/10.1093/biomet/30.1-2.81
7. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V.,
   Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S.,
   & Kiela, D. (2020). Retrieval-Augmented Generation for
   Knowledge-Intensive NLP Tasks. *NeurIPS*. arXiv:2005.11401.
8. Qwen Team (2024). *Qwen2.5 Technical Report*. arXiv:2412.15115.
9. Ray, S., Pan, R., Gu, Z., Du, K., Feng, S., Ananthanarayanan, G.,
   Netravali, R., & Jiang, J. (2025). METIS: Fast Quality-Aware RAG
   Systems with Configuration Adaptation. *SOSP '25*. arXiv:2412.10543.
10. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings
    using Siamese BERT-Networks. *EMNLP-IJCNLP*. arXiv:1908.10084.
11. Tawil, A. (2026). *David vs Goliath: agentic versus naive RAG across
    model scales*. Unpublished manuscript.
12. Xiao, S., Liu, Z., Zhang, P., Muennighoff, N., Lian, D., & Nie, J.-Y.
    (2023). *C-Pack: Packed Resources For General Chinese Embeddings*
    (BGE models). arXiv:2309.07597.

# Measurement 01 — Retrieval
### Day-by-day working plan · what to code, what to write

**Dates:** Mon 3 Aug → Sun 9 Aug 2026 · 6 measure + 1 write
**Paper:** *Does the Pareto-optimal RAG configuration transfer across GPU tiers?*
**Status:** Day 6 done — **Hypothesis 1: PARTIAL.** Frontier membership changed
(A100: 4 members, L4: 6, 3 shared), but the reranker is not the knob that
moved — `embed_model` is. Cross-encoder reranking sits on *both* cards'
frontiers. The specific prediction (reranker flips off, chunk/top-k
invariant) missed. Details: `PAPER.md` §4.

This is the execution plan for Block 1. Policy and rationale live in `REFERENCE.md`; this file holds the daily detail. Where the two disagree, `REFERENCE.md` wins.

---

## 1. Hypothesis 1 — committed before any sweep

> The Pareto-optimal RAG configuration is **not hardware-invariant**. Chunk size and top-k keep their optimal settings across GPU tiers, but cross-encoder reranking flips from Pareto-optimal on the A100 to Pareto-dominated on the L4 — its latency cost grows faster than the recall it buys.

| | |
|---|---|
| **Refuted if** | Both frontiers contain the same configurations in the same order. Identical composition means the recommendation transfers and the thesis fails in this setting. |
| **Partial if** | Frontier membership changes but the reranker is not the knob that moved. Report which knob did, and say plainly that the specific prediction missed. |
| **Never cut** | The reranker knob is the hypothesis. The L4 is the thesis. Everything else in the grid is negotiable. |
| **Write it first** | This paragraph goes into `SCOPE.md` and paper §1 on Mon 3 Aug, before a single cell runs. |

**Headline figure.** recall@5 versus p95 end-to-end latency, one series per card, frontier members marked, identical config IDs labelled on both curves. The prediction is that the reranker point sits *on* the A100 frontier and *under* the L4 frontier. If both land on their frontier, the hypothesis is refuted.

---

## 2. The configuration grid — frozen 3 Aug

The first three knobs determine the **index**; the last two are **query-time only**. That split is what makes 96 cells cost 12 index builds.

| Knob | Levels | Count | Affects |
|---|---|---|---|
| Chunk size | 256 / 512 / 1024 tokens | 3 | Index — rebuild required |
| Chunk overlap | 0 / 15% | 2 | Index — rebuild required |
| Embedding model | small (384d) / base (768d) | 2 | Index — rebuild required |
| Top-k | 3 / 5 / 10 / 20 | 4 | Query time only |
| Reranker | off / cross-encoder | 2 | Query time only — **the hypothesis** |

- **Indices to build:** 3 × 2 × 2 = **12**
- **Cells per card:** 12 × 4 × 2 = **96** — report this number in the paper
- **Total runs:** 192

### Cut order, if a sweep will not finish

1. Drop **overlap** → 48 cells
2. Drop **embedding model** → 24 cells
3. Reduce **top-k** to 3 / 10 / 20 → 18 cells

Decide before starting the sweep, never halfway through. Report the reduced grid honestly in Method.

**Never cut the reranker** — it is the hypothesis. **Never cut the L4** — it is the thesis. A single-card result is not a result.

---

## 3. Metrics

| Metric | Family | Definition, and why it is here |
|---|---|---|
| **recall@5** | Quality | Fraction of queries where ≥1 gold chunk appears in the top 5. **The headline y-axis.** |
| **recall@10** | Quality | Same at k=10. Distinguishes bad ranking from bad retrieval. |
| **MRR** | Quality | Mean reciprocal rank of the first gold chunk. Sensitive to ordering — which is exactly what a reranker changes. |
| **Context precision** | Quality | Fraction of returned chunks that are gold. Falls as top-k rises; this is the cost of a large k that recall hides. |
| **p50 / p95 latency** | Cost | End-to-end per query, over repeats. **Never the mean** — rented hosts have asymmetric tails. |
| **Per-stage latency** | Cost | embed · search · rerank · generate, separately. This is what lets you say *why* the reranker flipped, not just that it did. |
| **Peak VRAM** | Cost | `torch.cuda.max_memory_allocated()`, reset per cell. On a 24 GB L4 this is a live constraint. |
| **Tokens/sec** | Cost | Generation throughput. Separates retrieval cost from generation cost. |
| **Index build time** | Cost | Per index, recorded once. A config that retrieves fast but indexes for an hour is not free. |
| **OOM / refusal** | Result | Any cell the card cannot run. **A refusal is a finding**, not a gap. Record it; never silently skip. |

### Record schema — one JSON line per cell

```json
{
  "config":    {"chunk_size":512,"overlap":0.15,"embed":"base",
                "top_k":5,"reranker":"cross-encoder"},
  "config_id": "sha1:4f9c…",
  "quality":   {"recall@5":0.83,"recall@10":0.90,"mrr":0.71,"ctx_prec":0.34},
  "cost":      {"p50_ms":812,"p95_ms":1490,
                "stages_p50_ms":{"embed":9,"search":4,"rerank":121,"generate":678},
                "peak_vram_mb":9820,"tok_per_s":41.2,"index_build_s":73},
  "run":       {"n_queries":30,"repeats":3,"warmup":3,"seed":1337,"status":"ok"},
  "prov":      {"gpu":"NVIDIA A100 80GB PCIe","driver":"550.54.15","cuda":"12.4",
                "torch":"2.4.0","sbert":"3.0.1","faiss":"1.8.0",
                "git_sha":"a91f0c2","corpus_sha":"…","gold_sha":"…"}
}
```

`config_id` must match across cards. If it does not, the comparison is void.

---

## 4. Three decisions that are easy to get wrong

### 4.1 Gold labels cannot be chunk IDs

Chunk IDs only exist relative to one chunk size, and **three of the five knobs change the chunking**. A label made at 512 tokens is meaningless at 256 or 1024.

Label **character spans in the source document** instead — `{doc_id, char_start, char_end}` — then count a retrieved chunk as relevant if it overlaps a gold span by **≥ 50%** of that span. State the threshold in Method; it is a defensible choice only if it is stated.

Missing this on day 2 costs you day 4.

### 4.2 Order the sweep by index, not by cell

Cache indices on `(chunk_size, overlap, embed)` and run all 8 query configs against each before moving on. Sweeping in cell order rebuilds the index 96 times instead of 12 — the difference between a 4-hour sweep and a 12-hour one.

### 4.3 The generator model is a schedule decision

96 cells × 30 queries × 3 repeats ≈ **8,600 generations per card**. Use a **3B-class instruct model** with `max_new_tokens=128` and greedy decoding. A 7B doubles days 4 and 5 and buys nothing — the study is about retrieval configuration, not generation quality.

---

## 5. Day 1 — Mon 3 Aug · Scope freeze, then the baseline

Nothing is built before the scope is committed. Then the simplest RAG that works end to end: one hard-coded configuration that answers from the corpus and refuses when it cannot.

### Build

New files: `SCOPE.md` · `chunker.py` · `embed.py` · `store.py` · `pipeline.py` · `configs/grid.yaml`

- **`SCOPE.md` committed first.** One paragraph: in scope, out of scope, Hypothesis 1 and its refutation condition verbatim. Git commit before any other file exists.
- **Corpus.** 15 topic PDFs into `corpus/`. Record page count, total tokens, extraction library and version. Hash the directory into `corpus_sha`.
- **`chunker.py`** — recursive character splitting parameterised by `(size, overlap)`. Every chunk keeps `{doc_id, char_start, char_end}`, which is what makes day 2's labelling scheme work.
- **`embed.py`** — sentence-transformers behind one interface, two models registered as `small` / `base`. Pin both revisions.
- **`store.py`** — FAISS index plus a sidecar metadata table. Retrieval returns chunk text *and* its source span.
- **`pipeline.py`** — `answer(query, cfg)` returning `{answer, retrieved_chunks}`. Answer-only-from-context prompt, greedy decoding, `max_new_tokens=128`, seed fixed.
- **`grid.yaml`** — the full 96-cell grid, written today and not touched again.

### Write

- **§2 Related work — in full, ~350 words.** Depends on zero data and is already drafted in `REFERENCE.md` §4.1. One sentence each on syftr, RAG-Stack, RAGSmith, METIS and the cost–latency–quality benchmark line, closing on the gap sentence: *existing work locates the configuration frontier on fixed hardware; whether the frontier's composition is stable across hardware tiers has not been tested.*
- **§1 Introduction — the hypothesis paragraph only.** Practitioner problem, question, Hypothesis 1 with refutation condition. Contribution bullets wait for day 7.
- **§3.1 System under test.** Corpus stats, chunking method, both embedding models with revisions, vector store, the exact generation prompt, decoding settings.
- **§3.2 The configuration grid** as a table, with the 12-index / 96-cell decomposition explained.
- **`LOG.md` entry.** Raw notes: what ran, what broke, what you chose and why.

> **The number:** the baseline answers a question that is in the corpus, and refuses one that is not.

---

## 6. Day 2 — Tue 4 Aug · The gold set

The most consequential day of the block. Every number in the paper inherits the validity of these labels — which is why you make them by hand and why the procedure gets described rather than glossed.

### Build

New files: `data/gold.jsonl` · `eval.py` · `relevance.py` · `tests/test_eval.py`

- **30 questions, hand-labelled.** Each record: `{qid, question, gold_spans:[{doc_id,char_start,char_end}], notes}`. Spread across documents; include a few needing two documents and a few answerable from one sentence.
- **Label spans, never chunk IDs** (§4.1). Budget most of the day — 30 careful labels is four to five hours of honest work.
- **`relevance.py`** — `is_relevant(chunk_span, gold_span, thresh=0.5)` by overlap fraction. One function, one threshold, used everywhere.
- **`eval.py`** — recall@k, MRR, context precision from retrieved chunks plus the gold record. Pure function, no pipeline dependency, so it is unit-testable.
- **`test_eval.py`** — hand-built cases with known answers. An off-by-one in recall@5 should surface today, not on day 6.
- **Hash the gold set** into `gold_sha` and freeze it. Any later edit invalidates both cards.

### Write

- **§3.3 Gold set construction — the section a reviewer actually checks.** 30 queries, how sourced, the span-based labelling scheme, the 50% overlap rule and why, single annotator, hours spent, how ambiguous cases were resolved.
- **§3.6 Metric definitions.** recall@k, MRR and context precision written as formulas over spans, so "recall@5" is unambiguous.
- **§5 Threats — first draft of two entries.** Single annotator with no inter-annotator agreement figure; question distribution chosen by the same person who built the corpus. Write them honestly while the compromises are fresh.
- **`LOG.md` entry** — including every question that was hard to label. Those become the concrete examples in Threats.

> **The number:** `eval.py` scores any configuration handed to it, and the unit tests pass.

---

## 7. Day 3 — Wed 5 Aug · Instrument the stages

Wire Block 0's harness into the pipeline so one run of one config emits quality and cost as a single record. If any number still needs a note in a spreadsheet afterwards, the harness is not finished.

### Build

New: `run_cell.py` · `tests/test_record.py` — existing: `bench.py` · `provenance.py`

- **Per-stage timers** around embed · search · rerank · generate, with `torch.cuda.synchronize()` before each stop, or the GPU timings are fiction.
- **Repeats and warmup.** 3 warmup queries discarded, then 3 repeats per query. p50 and p95 computed over all 90 measurements per cell, not over per-query means.
- **Peak VRAM** — `reset_peak_memory_stats()` at cell start, `max_memory_allocated()` at end.
- **`run_cell.py`** — config in, one JSON line out, matching the schema in §3 exactly. Catches OOM and writes `status:"oom"` rather than crashing the sweep.
- **Provenance stamping** from Block 0: GPU, driver, CUDA, library versions, git SHA, seed, corpus and gold hashes.
- **Index cache** keyed on `(chunk_size, overlap, embed)` (§4.2).
- **Smoke test:** run 4 cells end to end and diff the records. Everything except `config` and timings should be identical.

### Write

- **§3.4 Harness and measurement methodology.** Timing method and synchronisation, warmup policy, repeat count, why p50/p95 and not the mean, VRAM measurement, how OOM is recorded rather than dropped.
- **§3.5 Hardware and software.** The A100 / L4 table with bandwidth figures. Leave driver and library versions as placeholders and fill them on days 4–5 from the actual provenance stamps — do not type them from memory.
- **§3.7 Reproduction.** The one command that regenerates a cell. Written now because it is true now; written on day 7 it will be approximately true.
- **`LOG.md` entry.**

> **The number:** one invocation produces a complete record — quality, cost and provenance — with no manual bookkeeping.

---

## 8. Day 4 — Thu 6 Aug · The full grid on A100

A measurement day. The code is done; today it runs. Babysit the sweep, watch the first records for anything implausible, and resist reading the results as a story.

### Build and run

New: `results/a100.jsonl` · `verify_grid.py` — existing: `sweep.py`

- **Rent the A100**, run the Block 0 dummy sweep first to confirm the harness travelled, then start the real one.
- **Order the sweep by index**, not by cell (§4.2).
- **Resumability is load-bearing today.** The sweep skips any `config_id` already present in the jsonl. Test it by killing the process once, deliberately, in the first ten minutes.
- **`verify_grid.py`** — asserts 96 distinct `config_id`s present, no duplicates, no `status:"error"`. Run before closing the machine.
- **Sanity-check the first three records:** recall should rise with top-k, context precision should fall, rerank latency should be non-trivial at k=20. If any is inverted, stop and fix — do not sweep on a broken metric.
- **Schedule risk.** ≈8,600 generations. If the projected finish is past midnight, apply the cut order *before* starting.

### Write

- **`LOG.md` entry** — wall-clock per cell, anything that failed, any level cut and why.
- **Fill the real version numbers** into §3.5 from the provenance stamp.
- **Method corrections only.** If the sweep forced a change, amend §3 today while you remember it.

> **Locked:** no results prose today. Logging a number is fine; explaining what it means is not.

> **The number:** a complete A100 grid — 96 cells, zero missing, `verify_grid.py` green.

---

## 9. Day 5 — Fri 7 Aug · The identical grid on L4

Same config space, same corpus, same gold set, same seed. The only permitted difference between the two result files is the provenance stamp. This is the day the thesis rests on.

### Build and run

New: `results/l4.jsonl` · `verify_pair.py` · `results/refusals.md`

- **Rent the L4.** Same container image, same pinned versions, same `grid.yaml`.
- **`verify_pair.py`** — asserts both jsonl files carry the identical set of `config_id`s, and identical `corpus_sha` and `gold_sha`. If this fails, the comparison is void and nothing downstream matters.
- **Expect OOM at 24 GB.** chunk 1024 × top-k 20 × cross-encoder × base embeddings is the likely casualty. Record each as `status:"oom"` with its config — **a refusal is a result**, and it is the practical half of the hardware-dependence claim.
- **`refusals.md`** — every cell the L4 declined, with the error. Becomes a table in §4.
- **Do not substitute a different card** if the L4 is unavailable. Wait. Card identity is load-bearing; a silent swap invalidates the paper.
- **Expect it to be slower.** Budget 2–3× the A100's sweep time. Start early.

### Write

- **`LOG.md` entry** plus the refusal log.
- **§3.5 completed** with the L4's real driver and library versions.
- **§3.8 Handling of unrunnable cells.** How OOM cells are treated in frontier computation — excluded from the L4 frontier, reported as a table. Decide and write it today, *before* you have seen whether it helps or hurts the hypothesis.
- **§5 Threats** — add "two cards is not a continuum" and "single seed".

> **Locked:** still no results prose. The sweep completes tonight; the reading of it is tomorrow's work.

> **The number:** two complete grids over one config space, `verify_pair.py` green, refusals logged.

---

## 10. Day 6 — Sat 8 Aug · The frontier, then the verdict

The data is in and frozen. Today is analysis: compute both frontiers, find out whether the composition changed, and write the verdict paragraph.

### Build

New: `pareto.py` · `analysis.py` · `figs/fig1.pdf` — existing: `plot.py`

- **`pareto.py`** — frontier membership maximising recall@5 and minimising p95 latency. Returns the member set per card.
- **Figure 1** — recall@5 vs p95 latency, one series per card, frontier members marked, identical config IDs labelled on both curves so a reader can trace one point across cards. **This figure is the paper.**
- **Frontier membership table** — every config, on-frontier per card, and the delta. The direct answer to the hypothesis.
- **Knob-by-knob analysis** — for each knob, which value appears in frontier configs on each card. This is what turns "the frontier changed" into "the reranker moved and chunk size did not".
- **Kendall τ** between the two cards' config rankings — one number summarising how much the ordering transferred. Useful in the abstract.
- **Per-stage attribution** — if the reranker flipped, show the rerank-stage latency ratio between cards next to the recall gain it bought. Mechanism is what makes the result credible.

### Write

- **§4 Results — unlocked today.** Figure 1 with caption, frontier membership table, refusal table, knob-by-knob paragraph.
- **The verdict paragraph.** One paragraph, plainly: confirmed, refuted or partial, and exactly which knob changed status. No hedging across three sentences what one sentence settles.
- **Set the hypothesis pill** on the master plan page.
- **`LOG.md` entry.**

> **The number:** the crossing, or its documented absence — plus the name of the knob that moved.

---

## 11. Day 7 — Sun 9 Aug · Paper 1, writing only

No code. A different mode of work, which is the point of the day. Sections 2 and 3 already exist from the week, so today is the four verdict-dependent sections plus assembly.

### Already done — assemble, do not rewrite

| Section | Written |
|---|---|
| §2 Related work | Day 1 |
| §3.1 System under test, §3.2 Grid | Day 1 |
| §3.3 Gold set, §3.6 Metrics | Day 2 |
| §3.4 Harness, §3.5 Hardware, §3.7 Reproduction | Day 3, versions filled days 4–5 |
| §3.8 Unrunnable cells | Day 5 |
| §4 Results, Figure 1, verdict | Day 6 |

**Reconcile:** read §3 against `LOG.md` and correct anything the sweep changed. This is the only editing pass §3 gets.

### Write today

- **Abstract, ~150 words.** Lead with the number, not the setup. The first sentence carries the finding.
- **§1 Introduction, ~400 words.** Complete it — practitioner problem → question → three contribution bullets, which you can only write now that you know what the contributions are.
- **§5 Threats, ~250 words.** Assemble the entries drafted on days 2 and 5, add anything day 6 exposed: one corpus, one embedding family, two cards is not a continuum, single seed, single annotator, rented hosts with noisy neighbours.
- **§6 Conclusion, ~150 words.** What a practitioner should do differently on Monday.
- **References**, 10–15 entries, identifiers verified against primary sources — not copied from `REFERENCE.md` §15, which is a to-locate list.
- **The length check.** No page cap, but for every section over its budget, name in one sentence the specific fact that earned the space. If you cannot, cut back.

> **The number:** a finished draft — one figure, two or three tables, a stated verdict, verified references.

---

## 12. Paper 1 — which section is written which day

*Does the Pareto-optimal RAG configuration transfer across GPU tiers?*
No page cap; the budgets below are a starting allocation, not a ceiling.

| Section | Words | Written | Depends on |
|---|---|---|---|
| Abstract | 150 | Day 7 | The verdict |
| 1. Introduction | 400 | Day 1 + 7 | Hypothesis day 1; contributions day 7 |
| 2. Related work | 350 | Day 1 | Nothing — drafted in `REFERENCE.md` §4.1 |
| 3. Method | 700+ | Days 1–5 | What you built, on the day you built it |
| 4. Results | 800+ | Day 6 | Both sweeps complete |
| 5. Threats | 250+ | Days 2, 5, 7 | Drafted as compromises are made |
| 6. Conclusion | 150 | Day 7 | The verdict |
| References | 10–15 | Day 7 | Verified against primary sources |

**Six of eight sections are finished before day 7 arrives.** That is the point of writing daily.

### The rule that protects daily writing

No sentence *interpreting* a result may be written before the sweep producing it is complete. Logging a number in `LOG.md` is fine; explaining what it means is not. Writing a verdict on day 4 commits you to a story before the data has finished arriving — Rule 1 failing through the back door.

---

## 13. Contingency — decided now, not at 11pm on day 5

| If | Then |
|---|---|
| The sweep will not finish in the day | Apply the cut order — overlap, then embedding model, then top-k levels. Decide before starting. Report the reduced grid in Method |
| The L4 OOMs on many cells | Not a problem — it is the practical half of the finding. Record every refusal, exclude those cells from the L4 frontier, give them a table in §4 |
| The hypothesis is refuted on day 4 | Good. Finish the L4 sweep anyway — a clean refutation needs the full grid, and it is the more interesting paper |
| The two frontiers look identical | That is the refutation condition. Check `verify_pair.py` passed, report Kendall τ ≈ 1, write it up as a transfer result. Papers 2 and 3 are unaffected |
| The gold set takes all of day 2 and is not done | Finish it on the morning of day 3 and compress instrumentation. Never ship a partial gold set |
| A card is unavailable | Wait. Do not substitute. Card identity is load-bearing |
| Day 7 arrives with the sweep incomplete | Write the paper anyway, on the cells you have, with the reduced grid stated honestly. 10 August does not move |

---

## 14. Open decisions

These are not yet settled and should be before 3 August.

- **Embedding model pair.** Candidates: `bge-small-en-v1.5` (384d) / `bge-base-en-v1.5` (768d), or the all-MiniLM / all-mpnet pair. Pick one family so the small/base contrast is dimension, not architecture.
- **Cross-encoder.** `bge-reranker-base` or `ms-marco-MiniLM-L-6-v2`. The lighter one makes the hypothesis harder to confirm, which is the honest choice.
- **Generator.** A 3B-class instruct model, decided on availability across both cards.
- **Corpus topic.** 15 PDFs on one subject retrieves differently from 15 unrelated ones. One subject is the harder and more realistic test.

---

*Companion to `m01-retrieval-plan.html`. Policy lives in `REFERENCE.md` — amend that file, not this one.*

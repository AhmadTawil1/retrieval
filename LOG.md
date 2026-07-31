# LOG

Raw notes, one block per day. Not prose — this feeds Threats to Validity later
(REFERENCE.md §10.1), so record what actually happened, not a cleaned-up version.

---

## Day 1 — Fri 31 Jul 2026

Ran M01-RETRIEVAL.md Day 1 directly (not Block 0 first — decided by Ahmad).

- Corpus: 15 PDFs already gathered under `corpus/` (one, `Pytorch (1).pdf`, had
  a filename with a space/parens — renamed to `PyTorch.pdf` for a stable doc_id
  before hashing). 72 pages, 35,751 tokens (`bge-base` tokenizer), `pypdf` 6.14.2.
  `corpus_sha = 2b3486e1671811722b6f256f04257667eef78ccd`.
- Open decisions from REFERENCE.md §14 settled:
  - Embeddings: `bge-small-en-v1.5` / `bge-base-en-v1.5` (same family, dimension-only contrast).
  - Reranker: `ms-marco-MiniLM-L-6-v2` — the lighter model, deliberately, since
    it makes Hypothesis 1 harder to confirm.
  - Generator: `Qwen/Qwen2.5-3B-Instruct` — considered a larger model given
    Colab Pro access, but the 3B constraint is about sweep-day schedule risk
    (~8,600 generations) and generation quality being out of scope, not GPU
    availability, so kept at 3B.
  - Corpus topic: practical GenAI/LLM-engineering docs (RAG, vLLM, LangGraph,
    LlamaIndex, Hugging Face, fine-tuning, agentic AI, multimodal, PyTorch,
    Docker, FastAPI, API, uv, GenAI systems) — one coherent subject.
- Dev machine has no local CUDA GPU (`torch.cuda.is_available() == False`).
  Day 1 build/smoke-test therefore runs on CPU; the real A100/L4 sweeps on
  days 4–5 are unaffected since those are rented cards.
- Model revisions pinned by resolving each Hub repo's current commit SHA on
  first load rather than hand-typing a hash — recorded in
  `configs/model_pins.yaml`.
- `configs/grid.yaml` generated programmatically (not hand-written) from
  `pipeline.Config` + a deterministic `config_id` (SHA1 over the sorted
  config), asserted to produce exactly 96 unique IDs before writing.
- Baseline smoke test (`pipeline.py __main__`): one hard-coded config
  (chunk_size=512, overlap=0.15, embed=small, top_k=5, reranker=off), one
  in-corpus question, one out-of-corpus question. **Passed**: in-corpus
  question ("What is retrieval-augmented generation?") answered correctly
  and grounded in the corpus; out-of-corpus question ("What is the capital
  of France?") refused with the exact configured refusal string. Day 1's
  number is met.
- Two real bugs hit and fixed while getting there, both worth remembering:
  1. `chunker._merge`'s overlap-trim step was a **silent infinite loop**: if
     the pending `current` atoms already fit entirely within the overlap
     token budget, trimming removed nothing, so retrying the same oversized
     next atom against the same unchanged `current` never terminated. Burned
     ~40+ minutes of wall-clock before being caught (looked like a slow
     tokenizer at first, wasn't). Fix: if trimming removes nothing, drop
     overlap entirely for that one boundary so `current` empties and the
     retry can make progress. Also replaced the O(n^2) linear-growth
     `_hard_slice` fallback with a binary search — unrelated to the hang
     (never triggered on this corpus, since no separator-free run exceeds
     ~66 chars) but a correctness/perf improvement regardless.
  2. `pipeline.generate()` — `tokenizer.apply_chat_template(..., return_tensors="pt")`
     returns a `BatchEncoding` (dict-like), not a bare tensor, on the installed
     `transformers` 5.14.1. `model.generate(inputs, ...)` failed on
     `inputs.shape`. Fixed by passing `return_dict=True` and calling
     `model.generate(**inputs, ...)`, the version-robust pattern.
- Dev machine has no CUDA GPU, so the two smoke-test generations ran on CPU —
  a few minutes total for ~256 tokens. Confirms the pipeline is correct; real
  per-cell latency numbers are meaningless until measured on the rented
  A100/L4 in days 4–5.

---

## Day 2 — Fri 31 Jul 2026

Ran M01-RETRIEVAL.md Day 2 same-day as Day 1 (compressed schedule — decided by
Ahmad, not the calendar date in the plan).

- Decided explicitly with Ahmad: the 30 gold questions/spans are hand-labelled
  by him alone, matching what "single annotator" in Threats actually claims —
  not AI-drafted-then-reviewed, which would be a different (weaker) process
  worth a different Threats entry.
- Built the tooling side: `relevance.py` (`is_relevant`, one overlap-fraction
  function, one threshold), `eval.py` (recall@k, MRR, context precision, plus
  gold-set load/validate/hash), `tests/test_eval.py` (17 cases, including the
  recall@k off-by-one at exactly rank k and the is_relevant boundary at
  exactly 0.5 — all pass).
- Built `label_helper.py` so spans don't have to be hand-counted: `find
  <doc_id> "<phrase>"` returns exact char_start/char_end against the same
  `pypdf`-extracted text the pipeline indexes (not the PDF's visual layout —
  confirmed these differ: e.g. RAG.pdf's extracted text has
  "indexsmallchunks" with no spaces, and some special characters extract as
  `�`. Worth knowing when reading extracted text to write questions/spans).
- `data/gold.example.jsonl` added as a schema template (placeholder values
  only, not real content) — `data/gold.jsonl` itself is Ahmad's to write.
- §3.6 Metric definitions and the structural half of §3.3 Gold set
  construction and §5 Threats (single annotator; question distribution
  chosen by the corpus-builder) written now, since they don't depend on
  which 30 questions get picked. The annotator-specific facts (hours spent,
  question distribution, ambiguous cases) are marked `[pending]` in
  `PAPER.md` until the labelling is actually done.
- **The number is not yet met**: `eval.py` scoring real configs and the gold
  set existing both wait on the 30 labels. `[outcome: pending]`

### Gold set — completed same day

- `data/GOLD-PLAN.md` written first: allocation only (which doc, which type,
  how many), no questions. Types A=7 (single-sentence answer), B=9 (paragraph),
  C=8 (lexically confusable across docs), D=6 (needs two docs).
- All 30 questions and span selections written by Ahmad. Two rounds: v1, review
  pass, v2.
- Review of v1 found: (a) two type-C labels wrong — q14 marked confusable with
  Hugging_Face when its own span names RAG, and q23 (docker-compose) marked
  confusable with uv, which has no multi-container content; (b) five questions
  with high content-word overlap against their own span (q03 62%, q27 53%,
  q12 and q11 50%, q30 47%). q03 used six of the span's terms in the span's
  order.
- **Why (b) matters, and it belongs in Threats**: a question phrased in its
  span's vocabulary is retrieved by near-lexical match at any configuration, so
  recall saturates across the grid and configs stop separating. A saturated
  gold set yields two identical frontiers — which reads as a refutation of
  Hypothesis 1 but is really a dead instrument. Fixed in v2 by describing the
  situation without naming the concept.
- I had also flagged q13 and q02 as vocabulary problems before measuring;
  overlap was 14% and 12%. Both fine, left alone. Recording the wrong call
  because the corrected list is what drove v2.
- v2 fixes verified. q23 replaced with a real Docker/uv collision
  (`RUN pip install -r requirements.txt` vs `uv pip install` / `uv sync`).
- **Span repair was mechanical, done by Claude; spans were chosen by Ahmad.**
  The quoted span text was cleaned-up reading rather than `pypdf` output, so 13
  of 36 did not resolve. Causes: code listings extract with line numbers
  interleaved (`3class Net ( nn . Module ) :`), identifiers extract
  letter-spaced (`t a r g e t _ m o d u l e s`), LaTeX emphasis eats spaces
  (`Stage 1 (retrieve):pull`, `newbehavior`). Resolver: whitespace-insensitive,
  then digit-tolerant, then `difflib` anchor with coverage reported. Every span
  verified against extracted text after resolution.
- Two spans were non-contiguous (q29[1], q30[1] — prose plus a code line from
  elsewhere in the document). Split into separate contiguous spans. 36 → 38.
- Final: 30 records, 38 spans, `validate_gold` 0 problems, all 15 docs covered,
  span length min 38 / median 193 / max 414 chars.
- `gold_sha` = `3afa042d8d9cd6784e8e1b049e4f4f6c1d45709c` — frozen. Any edit
  from here invalidates both cards.
- **Open item for §5**: the gold set is constructed, not sampled — 8/30 written
  to be confusable, 6/30 needing two documents, both higher than natural query
  traffic would give. Draft paragraph in `data/GOLD-PLAN.md` §5; paste into
  `PAPER.md` before day 7.
- `[outcome: met]` — the gold set exists and `eval.py` can score against it.

---

## Day 3 — Fri 31 Jul 2026

Compressed schedule continues (Day 3 same session as Days 1–2).

- `bench.py` and `provenance.py` are Block-0 files per M01-RETRIEVAL.md, but
  Block 0 was skipped — built both fresh today instead of assuming they
  existed.
- Refactored `pipeline.py`: `retrieve()` split into named `embed_query` /
  `search` / `rerank` stages (previously inlined), so `run_cell.py` can time
  each independently; `generate()` now returns `(text, n_new_tokens)` for
  `tok_per_s`. Re-ran the Day 1 baseline smoke test after the refactor —
  identical in-corpus/refusal output, so behavior didn't change.
- `run_cell.py`: index cache keyed on `(chunk_size, overlap, embed_model)`;
  warmup = first 3 gold queries run once and discarded; timing = all 30
  queries × 3 repeats = 90 measurements/cell; OOM caught and recorded as
  `status:"oom"` rather than crashing. `validate_record` added for schema
  shape-checking (used by `tests/test_record.py`, not by the sweep itself).
- 28/28 unit tests pass (`test_eval.py` + new `test_record.py`, the latter
  covering `bench.py` percentiles and `validate_record` — all against fake
  data, no model loading, runs in seconds).
- Smoke test: 4 cells, 2 distinct indices (256/0.0/small and
  512/0.15/small) × {reranker off, cross-encoder}, `--n-queries 2 --warmup 1
  --repeats 1` (not the real 30/3/3 — this machine is CPU-only and a
  reduced protocol is enough to check the harness is wired correctly; days
  4–5 use the real numbers on rented cards). **Passed**: `run` and `prov`
  blocks byte-identical across all 4 records; index cache correctly reused
  (`index_build_s: 0.0`) on the 2 cells sharing an already-built index;
  rerank stage cost scales with over-fetch size (~0.9s at top_k=3→12
  candidates, ~2.9s at top_k=5→20) and measures ~0 when `reranker=off`,
  confirming the per-stage timers are honest, not fiction.
- **CPU generation is the bottleneck, and it is context-size sensitive, not
  constant**: `chunk_size=256/top_k=3` cells generated in ~118–156s per
  call; `chunk_size=512/top_k=5` cells (≈3x more context tokens) took
  ~410s per call — a 2.6–3.5x slowdown tracking context growth, on a
  no-GPU dev box (`torch.cuda.is_available() == False`). Total smoke-test
  wall time ≈ 45 minutes. Confirms the pipeline is correct; these numbers
  say nothing about real per-cell latency, which only rented A100/L4
  hardware in days 4–5 can produce. Discussed moving today's dev/smoke work
  to Colab+A100 for velocity (rough estimate: 50–100x faster generation);
  decided to keep the actual Measurement 01 sweeps on rented cards per
  REFERENCE.md (card identity must be load-bearing and deliberately chosen,
  which a Colab-allocated GPU doesn't guarantee).
- `[outcome: met]` — one invocation (`run_cell.py <config_id>`) produces a
  complete record: quality, cost, and provenance, no manual bookkeeping.

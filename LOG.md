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

---

## Day 4 — the full grid on A100

- Card: **Colab Pro, A100 runtime, confirmed `NVIDIA A100-SXM4-40GB`** via
  `torch.cuda.get_device_name(0)` before any real work started. Colab Pro
  lets the runtime type be selected explicitly (A100/L4/T4/CPU/TPU all
  listed), which resolves the card-identity concern raised before starting
  — this is a deliberate choice, not an unpredictable allocation.
- **Correction, not a deviation**: the plan assumed A100 **80GB** (~2,039
  GB/s); the card actually available is the **40GB SXM4** SKU (~1,555
  GB/s). Corrected `PAPER.md` §3.5 to the real card rather than leaving the
  planning assumption in place. Bandwidth ratio vs the L4 is therefore
  ≈5.2×, not ≈6.8× — still a wide enough spread for the study's purpose.
  40GB VRAM is not expected to constrain this workload (unlike the L4's
  24GB, where OOMs are an expected, meaningful result).
- Output written to a Google-Drive-mounted path
  (`/content/drive/MyDrive/retrieval_results/a100.jsonl`), not Colab's local
  disk — local disk is ephemeral and a disconnect would otherwise lose
  everything resumability is supposed to protect.
- Ran `sweep.py --output <drive path>` with the real protocol (30
  queries × 3 repeats, no `--limit`/reduced flags — first time this
  machine has run the full parameters). First-run cost: model downloads
  (embedding model, reranker, Qwen2.5-3B-Instruct ~6GB) all completed in
  under 30s on Colab's network — much faster than expected, not a concern.
- `verify_grid.py data/a100.jsonl --sanity`: 96/96 config_ids present, no
  duplicates, no bad status — GREEN. Sanity checks behave as expected:
  recall@5 rises with top_k then plateaus after k=5 (0.799→0.889→0.889→0.888),
  ctx_prec falls (0.284→0.198), rerank stage at top_k=20 costs a real 97.9ms
  p50 (not a no-op). Two provenance incidents surfaced and closed while
  reviewing the records — one false alarm, one real gap — detailed below.
  `[outcome: met]` — 96/96 valid records, full protocol, real A100 card.

### False alarm — `corpus_sha` mismatch, resolved. No re-run needed.

- A session reviewing the A100 records reported that their `corpus_sha`
  (`a2a9aecc…`) did not match the local corpus (`2b3486e1…`), and inferred the
  Colab PDFs were byte-different from the committed ones — which would have
  meant the gold spans described a different document set than the one measured.
  That inference was wrong and the conclusion is the reverse.
- Checked directly: `a2a9aecc…` is the hash of **both** the working tree and the
  git blobs at `HEAD:corpus/*` — verified by hashing `git cat-file blob` output
  rather than files on disk. Zero files differ between worktree and blob. **The
  A100 records are correct and the Colab corpus was byte-identical to the repo.**
- `2b3486e1…` reproduced exactly, two ways: hashing in unsorted `glob()` order,
  or hashing in case-insensitive filename order. Both give the same result here
  because the corpus lives on a Windows-backed filesystem, whose directory
  enumeration is case-insensitive. So the reviewing session did not call
  `chunker.corpus_sha` — it reimplemented the hash and dropped the `sorted()`.
- Why the order matters: Python's `sorted()` is ASCII-ordinal, so uppercase
  precedes lowercase and `uv.pdf` / `vLLM.pdf` sort **last**. Case-insensitive
  order puts `Agentic_AI` before `API` and `LLM` after `LangGraph`/`LlamaIndex`.
  Same bytes, same files, different concatenation order, different SHA1.
- Also ruled out while checking: CRLF/LF filter mangling (repo has no
  `.gitattributes`, so it was a live hypothesis — both conversions produce
  hashes matching nothing), and extra or missing files in `corpus/`.
- Audited every corpus enumeration in the repo for the same bug:
  `chunker.py:150,159,178` and `label_helper.py:26,33` all sort;
  `eval.py:92` globs unsorted but builds a `set`, where order is irrelevant.
  **No latent ordering bug in the codebase.** The defect was only in the
  ad-hoc verification, not in anything that produced data.
- **Lesson worth keeping**: provenance checks must call the same function that
  wrote the field. A reimplemented check compares two different definitions and
  reports a data problem that does not exist. This one nearly cost a full re-run
  day out of a 28-day schedule.
- Day 4 data stands. Proceed to Day 5 (L4) with `corpus_sha = a2a9aecc…` as the
  value `verify_pair.py` must find on both sides.

### Real gap — model revisions not recorded for the A100 sweep

- Unlike the `corpus_sha` alarm, this one is genuine. `provenance.stamp()`
  recorded the `sentence_transformers` package version but no model revisions,
  so `data/a100.jsonl` does not say which model weights produced the numbers.
- **Worse than first reported.** The review flagged `base` only. Audit found
  `pipeline.py` loaded the reranker as `CrossEncoder(RERANKER_MODEL_ID)` and the
  generator as `from_pretrained(GENERATOR_MODEL_ID)` — no revision argument, and
  neither went through the pin mechanism at all. Three of four models
  unrecorded, one of them the cross-encoder, which *is* Hypothesis 1.
- **Better than first reported for `small`.** `configs/model_pins.yaml` is
  tracked, and `git show 04caf49:configs/model_pins.yaml` (04caf49 being the
  `git_sha` on all 96 records) gives `small: 5c38ec7c…`. Colab cloned that
  commit, so `_load_pins()` returned it. That one is a measurement, not a guess.
- Colab VM was disconnected and deleted before the runtime-written `base` pin
  could be read. **Direct evidence for `base` is unrecoverable.**
- Fallback used: resolved `main` on 1 Aug 2026 via the HF API. Defensible
  because each repo's last update predates the 31 Jul 2026 sweep —
  bge-base 21 Feb 2024, ms-marco-MiniLM-L6-v2 29 Aug 2025,
  Qwen2.5-3B-Instruct 25 Sep 2024 — so `main` was static across the window.
  Assumes no force-push or reverted commit. **These three are inferences and are
  labelled as such in `configs/model_pins.yaml`; §3.1 and §5 must say so too.**
- Fix, so it cannot recur on L4:
  - new `pins.py` — one `revision_for(model_id)` used by every model load,
    keyed by model ID rather than short name, with a legacy fallback so the
    tracked `small:` line still resolves;
  - `pipeline.py` passes `revision=` to the reranker, the generator and its
    tokenizer;
  - `embed.py` delegates to `pins.py` instead of its own copy;
  - `provenance.stamp()` emits `prov.model_revisions` — all four models, every
    record. A pin that lives only in a file can die with the machine; a pin in
    the record cannot.
  - `configs/model_pins.yaml` pre-seeded with all four so the L4 run does not
    re-resolve `main`.
- Verified: all four resolve from the file with no network call, and reading
  them does not rewrite the file. `py_compile` clean on all four modules.
- **Consequence for the paper — `git_sha` will differ between cards.** A100 ran
  at 04caf49; L4 will run at the commit containing this fix. Method must state
  both and note the delta is provenance capture only, not the measured path
  (`git diff 04caf49 -- chunker.py store.py eval.py relevance.py` is empty).
  Chose this over leaving three of four models unrecorded on both cards.
- **Lesson, and it is the same shape as the corpus_sha one**: provenance that
  lives anywhere other than the record is not provenance. `model_pins.yaml`
  looked like a pin file but was a cache — written at runtime, on whichever
  machine happened to load first, and never committed back.

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

# Gold set allocation plan — 30 questions

Structure only. The questions and the spans are yours to write; this file decides
*how many of what kind, and against which documents*, so the distribution is a
design decision made in advance rather than whatever happened to come to mind.

Work through the table, fill `data/gold.jsonl` as you go, and record the two
things §3.3 needs but cannot be reconstructed later: **hours spent** and **which
questions were ambiguous**.

---

## 1. Why the mix matters

This is the part worth reading before writing question one. The composition of
the gold set decides whether your knobs are separable at all.

**Answer length pulls chunk size in opposite directions.** A question whose
answer is one sentence is best served by a 256-token chunk — a 1024-token chunk
buries it in noise and drags context precision down. A question whose answer is
spread over a paragraph is the reverse: small chunks split the answer and recall
drops. If all 30 questions had one-sentence answers, chunk size would not be a
trade-off at all — 256 would dominate everywhere, and that knob would produce a
single winner instead of a frontier. The mix is what makes chunk size a real
axis.

**Confusable questions are what the reranker exists for.** A cross-encoder earns
its latency only when the first retrieval pass returns chunks that look right and
are not. If every question is lexically unambiguous, the base retriever already
gets it, the reranker adds latency for no recall, and it is Pareto-dominated on
*both* cards. That refutes Hypothesis 1 — but for an uninteresting reason: not
because the recommendation transfers, but because the corpus never tested the
knob. **The 8 type-C questions are load-bearing for the hypothesis.** Your corpus
helps here: RAG / LlamaIndex, LangGraph / Agentic_AI and vLLM / LLM all discuss
overlapping vocabulary.

**Multi-document questions are what makes top-k a real axis.** A question with a
single gold span is often satisfied at k=3, so k=5, 10 and 20 all score the same
and the top-k axis flattens. A question with gold spans in two documents cannot
reach full recall at low k. Six of these keep the axis alive.

---

## 2. Question types

| Type | Count | What it is | Which knob it stresses |
|---|---|---|---|
| **A** | 7 | Answer is a single sentence. Gold span is short — roughly one line of extracted text. | Chunk size (small wins), context precision |
| **B** | 9 | Answer is spread over a paragraph or a short section. Gold span is longer. | Chunk size (large wins), overlap |
| **C** | 8 | Vocabulary appears in two or more documents; the answer is only correct in one. | **The reranker — the hypothesis** |
| **D** | 6 | Needs two documents. Two gold spans, different `doc_id`s. | Top-k |

---

## 3. The allocation

Every document appears at least once. Fill the question column as you go.

### Single-document — q01 to q24

| qid | Doc | Type | Confusable with | Your question |
|---|---|---|---|---|
| q01 | RAG | C | LlamaIndex | |
| q02 | RAG | B | — | |
| q03 | LlamaIndex | C | RAG | |
| q04 | LlamaIndex | A | — | |
| q05 | LangGraph | C | Agentic_AI | |
| q06 | LangGraph | B | — | |
| q07 | Agentic_AI | C | LangGraph | |
| q08 | Agentic_AI | B | — | |
| q09 | vLLM | C | LLM | |
| q10 | vLLM | A | — | |
| q11 | LLM | B | — | |
| q12 | LLM | A | — | |
| q13 | Fine_Tuning | B | — | |
| q14 | Fine_Tuning | C | Hugging_Face | |
| q15 | Hugging_Face | A | — | |
| q16 | PyTorch | B | — | |
| q17 | Multimodal | A | — | |
| q18 | GenAI_systems | B | — | |
| q19 | FastAPI | C | API | |
| q20 | FastAPI | A | — | |
| q21 | API | B | — | |
| q22 | Docker | A | — | |
| q23 | Docker | C | uv | |
| q24 | uv | B | — | |

### Multi-document — q25 to q30

Two gold spans, one in each document. Write the question so that **neither
document alone is sufficient** — if one of them answers it fully, it is a type-B
question with a redundant span, and it will not stress top-k.

| qid | Docs | Your question |
|---|---|---|
| q25 | RAG + LlamaIndex | |
| q26 | LangGraph + Agentic_AI | |
| q27 | vLLM + LLM | |
| q28 | Docker + uv | |
| q29 | FastAPI + API | |
| q30 | PyTorch + Fine_Tuning | |

### Coverage check

| Doc | Questions | | Doc | Questions |
|---|---|---|---|---|
| RAG | 3 | | LLM | 3 |
| LlamaIndex | 3 | | Fine_Tuning | 3 |
| LangGraph | 3 | | vLLM | 3 |
| Agentic_AI | 3 | | Docker | 3 |
| FastAPI | 3 | | API | 2 |
| PyTorch | 2 | | uv | 2 |
| Hugging_Face | 1 | | Multimodal | 1 |
| GenAI_systems | 1 | | | |

36 document-appearances across 30 questions. All 15 documents covered.

---

## 4. Procedure

**Write the question before you find the span.** Read the section, understand it,
close it, then write the question in your own words. If you write the question
while looking at the sentence, you will reuse its vocabulary — and a question
phrased in the document's own words turns retrieval into string matching. That
inflates recall uniformly across the grid and compresses the very spread your two
frontiers are made of.

**Then find the span:**

```
uv run python scripts/label_helper.py dump RAG            # read with offset markers
uv run python scripts/label_helper.py find RAG "phrase"   # exact char_start / char_end
```

**Remember the extraction quirks** already logged: the indexed text is `pypdf`
output, not the PDF's visual layout. `RAG.pdf` contains `indexsmallchunks` with
no spaces, and some characters extract as `�`. Search for a fragment you have
confirmed exists in the *extracted* text, not what you see in a PDF reader.

**Span boundaries.** The span should cover the text that actually answers the
question — not the whole section, not a fragment of a sentence. `is_relevant`
counts a chunk as relevant at ≥50% overlap of the gold span, so an over-wide
span makes the question artificially easy and an over-narrow one makes it
brittle.

**The `notes` field** is not decoration — it is the raw material for §3.3 and
§5. Record the type, and anything you hesitated over:

```
"notes": "type C — 'chunking strategy' also appears in LlamaIndex.pdf §3;
          only RAG.pdf gives the token-count guidance. Hesitated between
          this span and the table above it."
```

**Validate as you go**, rather than at the end:

```
uv run python -c "import eval, pathlib; \
  print(eval.validate_gold(eval.load_gold(pathlib.Path('data/gold.jsonl')), pathlib.Path('corpus')))"
```

**Record the clock.** Note your start and stop times. §3.3 claims a single
annotator; "roughly five hours across one day" is the kind of concrete detail
that makes that claim credible.

---

## 5. Threats entry this plan creates

Write this into §5 while it is fresh — the deliberate design is defensible, but
only if it is disclosed:

> The query set was constructed to a planned distribution rather than sampled
> from observed usage: 8 of 30 questions were written to be lexically confusable
> across documents, and 6 require evidence from two documents. Both proportions
> are higher than a naturally-occurring query distribution would likely produce,
> and both favour configurations that rerank and configurations that retrieve at
> higher k. The distribution was fixed before any measurement, but it does mean
> absolute recall figures are not comparable to other corpora — only the relative
> ordering of configurations within this study is.

That paragraph costs you nothing and pre-empts the first question a reviewer
would ask about a hand-built gold set.

---

*Companion to `M01-RETRIEVAL.md` §6. The labels themselves are hand-made by the
single annotator — see `LOG.md`, Day 2.*

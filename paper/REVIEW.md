# Review of PAPER.md — 2 Aug 2026

Everything below was checked against `data/a100.jsonl`, `data/l4.jsonl`,
`scripts/analysis.py` and the primary sources. Numbers I could not reproduce are
marked, with what I got instead.

---

## 1. Must fix — one factual error

### §3.1 reports the wrong `corpus_sha`

PAPER.md states:

> `corpus_sha` (SHA1 over sorted filename+bytes pairs): `2b3486e1671811722b6f256f04257667eef78ccd`

**All 192 records say `a2a9aecc051466d25ff4f3e48aaf2c69aba1a8aa`** — verified as
the unique value across both files, and equal to a fresh hash of both the
working tree and the git blobs at `HEAD:corpus/*`.

`2b3486e1…` is the artifact hash from the Day 4 false alarm — the value produced
by hashing in unsorted/case-insensitive filename order. It has leaked from the
incident into the paper.

This is the most damaging error in the draft, because §3.4 and the first
contribution bullet both sell provenance discipline as a contribution. A
reviewer who runs `chunker.corpus_sha` gets a different number than the paper
reports, in the paper's own provenance section. Fixed in the LaTeX.

---

## 2. Should fix — numbers that need their definition stated

### §4 and Abstract: the 1.40× rerank ratio is definition-dependent

`analysis.py:119` computes `stage_ratio` as a **ratio of means**
(mean L4 / mean A100), not a median of per-cell ratios. Both are defensible; they
disagree substantially here:

| Statistic | rerank stage, L4/A100 |
|---|---|
| Ratio of means (what the paper reports) | **1.403** |
| Median of per-cell ratios | **1.249** |
| Per-cell spread | min 1.13, p25 1.19, med 1.25, p75 1.42, max 1.65 |

The gap means the headline figure is pulled up by the slower cells. The claim
survives either way — 1.25× still exceeds every other stage — but the paper
should say which statistic it is, and ideally give both. A reviewer recomputing
this the obvious way will get 1.25 and wonder.

Also note an inconsistency in the comparison base: `rerank_stage_ratio` is
computed over the 48 reranker-on cells, while `embed`/`search`/`generate` ratios
are over all 96. Harmless for stages present in every cell, but worth one clause.

### "1.04–1.17× for every other pipeline stage" understates the range

Measured: embed **1.043**, generate **1.169**, search **1.000**. The lower bound
is 1.00, not 1.04 — search is the flat one, and it is the most interesting of the
three because it is flat *for a reason* (FAISS search is CPU-bound). Writing
"1.00–1.17×" is both accurate and a better argument.

### "rerank is ~1–1.5% of a cell's p95" is optimistic

Measured over the 48 reranker-on A100 cells: **median 0.81%, range 0.32–2.16%**.
The real range is wider in both directions than stated. Suggested wording:
"under 1% of a typical cell's p95 (median 0.81%, range 0.32–2.16%)". This makes
the "nowhere large enough to act" argument *stronger*, not weaker.

### §4 overall p95 ratio

Paper says 1.14× "across shared configs". `analysis.py` reports 1.135 over all 96
shared configs — correct as written. But over the 3 shared *frontier* configs it
is 1.078. Since the sentence sits next to frontier discussion, name the set.

---

## 3. Verified correct — no change needed

| Claim | Status |
|---|---|
| A100 frontier 4, L4 frontier 6, 3 shared, 7 union | ✅ reproduced exactly |
| All 7 frontier-table rows (configs, recall@5, both p95 values) | ✅ every cell matches |
| Kendall tau 0.802 | ✅ 0.8018 |
| Reranker recall@5 gain +0.018, identical on both cards | ✅ 0.0181 both |
| Zero OOMs on either card | ✅ 96/96 `status:"ok"` both |
| `config_id` sets identical across cards | ✅ |
| `gold_sha` 3afa042d… identical across cards | ✅ |
| Corpus 72 pages | ✅ |
| Software table (driver, CUDA, torch, sbert, faiss, git_sha) | ✅ matches both `prov` blocks |
| A100 records lack `model_revisions`; L4 records have it | ✅ exactly as §3.1 describes |
| 30 questions, 38 gold spans, span length 38–414, median 193 | ✅ |
| embed 1.04×, search ~1.00× | ✅ |

The `embed_model` swap story holds up: `93ce410a3c36` and `0ca6aa10c184` differ
only in `embed_model`, both at recall 0.933, and the frontier slot genuinely
swaps. The 9ms/44ms margins are real and the paper is right to flag them as
within host-noise scale rather than oversell them. That self-limiting paragraph
in §4 is the strongest writing in the draft.

---

## 4. Citations

Verified against arXiv abstract pages:

| Ref | ID | Status |
|---|---|---|
| syftr — Conway, Dey, Hackmann, Hausknecht, Schmidt, Steadman, Volynets | 2505.20266 | ✅ author list exact. **Add venue: AutoML 2025** — the arXiv page lists it |
| RAG-Stack — Jiang | 2510.20296 | ✅ title and content match; abstract explicitly spans "software to hardware", which supports your characterisation |
| RAGSmith | 2511.01386 | ✅ correct paper; abstract confirms evolutionary search and domain sensitivity |
| MS MARCO, Lewis et al., Sentence-BERT, Faiss, C-Pack, Qwen2.5 | as listed | ✅ standard identifiers, correct |

**One left to check yourself:** METIS (Ray et al., arXiv:2412.10543) is cited as
*SOSP '25*. I confirmed the arXiv ID is plausible but did not verify the venue
line or the 8-author list. Since §2 names it specifically, confirm the venue
before submission.

**Related-work framing is sound.** The gap sentence is properly earned: all four
systems locate a frontier on fixed hardware, and RAG-Stack is correctly described
as closest-but-not-the-same. No citation is doing work it cannot support.

---

## 5. Writing

**Strengths.** The measured/inferred distinction in §3.1 with the † footnote is
unusually honest and will read well. §4's refusal to oversell a 9ms margin, and
§5's admission that the lexical-saturation review had no independent check, are
the two places a reviewer decides you are trustworthy. Keep both verbatim.

**Weaknesses, in order:**

1. **The verdict is buried.** The abstract's first sentence is background. Lead
   with the finding: frontier membership changed, but not via the predicted knob.
2. **"§4.1"/"§4.2" cross-references in §3.3 and §3.4 point at `M01-RETRIEVAL.md`,
   not at this paper's §4.** In the paper they resolve to Results, which is
   wrong. Fixed in the LaTeX by removing them.
3. **Contribution 3 is a paragraph doing the work of a sentence.** Tightened.
4. **§3.4's tone drifts into lab-notebook** ("which is itself a check that the
   instrumentation is honest"). Good instinct, slightly informal register.
5. **Threats is long and unranked.** Seven entries of roughly equal weight. The
   inferred-reranker-revision one is the most serious — it touches an arm of the
   hypothesis — and should come first, not fourth.

---

## 6. What I changed in the LaTeX

- `corpus_sha` corrected to `a2a9aecc…`
- Stage-ratio definition stated explicitly; both statistics given
- Stage range corrected to 1.00–1.17×
- Rerank share corrected to median 0.81%, range 0.32–2.16%
- Abstract reordered to lead with the finding
- Threats reordered, most serious first
- Broken `§4.1`/`§4.2` cross-references removed
- syftr venue added
- METIS venue left as-is with a `% VERIFY` comment

# SCOPE — Measurement 01: Retrieval

**In scope.** One RAG pipeline (recursive-character chunking → sentence-transformer embedding → FAISS retrieval → optional cross-encoder rerank → 3B-class instruct generation) swept over the frozen 96-cell grid (chunk size × overlap × embedding model × top-k × reranker), measured identically on an A100 and an L4, against one 15-document corpus and one 30-query hand-labelled gold set. The output is two Pareto frontiers (recall@5 vs p95 latency) and a verdict on whether their composition matches.

**Out of scope.** Generation quality as an outcome (the generator is fixed and unswept — this is a retrieval-configuration study, not a generation study). The T4 third tier. Agentic or multi-hop retrieval. Any corpus other than the one frozen here. Any grid knob not in the table below.

**Hypothesis 1 — committed before any sweep.**

> The Pareto-optimal RAG configuration is **not hardware-invariant**. Chunk size and top-k keep their optimal settings across GPU tiers, but cross-encoder reranking flips from Pareto-optimal on the A100 to Pareto-dominated on the L4 — its latency cost grows faster than the recall it buys.

**Refuted if.** Both frontiers contain the same configurations in the same order. Identical composition means the recommendation transfers and the thesis fails in this setting.

**Partial if.** Frontier membership changes but the reranker is not the knob that moved. Report which knob did, and say plainly that the specific prediction missed.

**Never cut.** The reranker knob is the hypothesis. The L4 is the thesis. Everything else in the grid is negotiable.

*Verbatim from `M01-RETRIEVAL.md` §1. Where this file and that one disagree on scope, `M01-RETRIEVAL.md` — and above it, `REFERENCE.md` — win.*

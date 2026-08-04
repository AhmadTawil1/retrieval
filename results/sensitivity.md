# Sensitivity analysis — a100.jsonl vs l4.jsonl

96 configurations present on both cards.

## Frontier delta under p50 vs p95

| statistic | A100 | L4 | shared | frontiers identical? | reported swap present? |
|---|---|---|---|---|---|
| p95_ms | 4 | 6 | 3 | no | yes |
| p50_ms | 6 | 7 | 4 | no | no |

## Frontier delta under epsilon-dominance (p95)

| epsilon (ms) | A100 | L4 | shared | frontiers identical? |
|---|---|---|---|---|
| 0 | 4 | 6 | 3 | no |
| 10 | 7 | 8 | 4 | no |
| 25 | 10 | 9 | 6 | no |
| 50 | 15 | 13 | 10 | no |
| 75 | 23 | 18 | 15 | no |
| 100 | 29 | 26 | 21 | no |
| 150 | 46 | 34 | 29 | no |
| 200 | 59 | 44 | 42 | no |
| 300 | 73 | 52 | 52 | no |

## Configs on exactly one card's frontier (p95_ms)

| config_id | chunk | overlap | embed | top_k | reranker | recall@5 | card |
|---|---|---|---|---|---|---|---|
| sha1:0ca6aa10c184 | 1024 | 0.15 | base | 3 | cross-encoder | 0.933 | L4 only |
| sha1:1da8ed40f4d1 | 256 | 0.0 | small | 3 | cross-encoder | 0.700 | L4 only |
| sha1:8edd4990fff0 | 256 | 0.0 | small | 3 | off | 0.667 | L4 only |
| sha1:93ce410a3c36 | 1024 | 0.15 | small | 3 | cross-encoder | 0.933 | A100 only |

## Configs on exactly one card's frontier (p50_ms)

| config_id | chunk | overlap | embed | top_k | reranker | recall@5 | card |
|---|---|---|---|---|---|---|---|
| sha1:784968f9e90a | 512 | 0.0 | base | 3 | cross-encoder | 0.800 | L4 only |
| sha1:7854342500f9 | 1024 | 0.0 | base | 5 | off | 0.967 | L4 only |
| sha1:8fe126313da5 | 256 | 0.0 | base | 5 | cross-encoder | 0.800 | A100 only |
| sha1:e277b7b82305 | 512 | 0.0 | small | 3 | cross-encoder | 0.767 | L4 only |
| sha1:ee95533f568b | 1024 | 0.15 | base | 10 | off | 0.967 | A100 only |

# Day 6 analysis — data\a100.jsonl vs data\l4.jsonl

## Frontier delta (configs on either card's frontier)

| config_id | chunk_size | overlap | embed_model | top_k | reranker | recall@5 | A100 p95 | L4 p95 | on A100 | on L4 | delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sha1:0ca6aa10c184 | 1024 | 0.15 | base | 3 | cross-encoder | 0.933 | 5483ms | 5832ms | False | True | L4 only |
| sha1:1da8ed40f4d1 | 256 | 0.0 | small | 3 | cross-encoder | 0.700 | 5298ms | 5623ms | False | True | L4 only |
| sha1:71d5acecfa94 | 256 | 0.0 | base | 5 | off | 0.900 | 5329ms | 5672ms | True | True | both |
| sha1:7854342500f9 | 1024 | 0.0 | base | 5 | off | 0.967 | 5555ms | 6048ms | True | True | both |
| sha1:8edd4990fff0 | 256 | 0.0 | small | 3 | off | 0.667 | 5280ms | 5621ms | False | True | L4 only |
| sha1:93ce410a3c36 | 1024 | 0.15 | small | 3 | cross-encoder | 0.933 | 5474ms | 5876ms | True | False | A100 only |
| sha1:9df54f160c03 | 256 | 0.0 | base | 3 | off | 0.833 | 5249ms | 5659ms | True | True | both |

## Knob-by-knob frontier composition

| knob | A100 frontier values | L4 frontier values | changed? |
|---|---|---|---|
| chunk_size | [1024, 256] | [1024, 256] | False |
| overlap | [0.0, 0.15] | [0.0, 0.15] | False |
| embed_model | ['base', 'small'] | ['base', 'small'] | False |
| top_k | [3, 5] | [3, 5] | False |
| reranker | ['cross-encoder', 'off'] | ['cross-encoder', 'off'] | False |

## Kendall tau (latency ordering, A100 vs L4, n=96)

tau_b = 0.8018

## Reranker attribution

- mean recall@5 gain from reranker, A100: 0.0181
- mean recall@5 gain from reranker, L4: 0.0181
- rerank-stage p50 ratio (L4/A100): 1.403
- overall p95 ratio (L4/A100), all shared configs: 1.135
- embed-stage p50 ratio (L4/A100): 1.043
- search-stage p50 ratio (L4/A100): 1.000
- generate-stage p50 ratio (L4/A100): 1.169

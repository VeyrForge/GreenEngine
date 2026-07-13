# Benchmarks

Performance data for Green Engine. **Measured** = run on a reference machine or via committed
benchmark binaries. **Projected** = derived from measured hit-rates and hardware specs.

Reference machine: RTX 5060 Ti 16 GB, 32-core x86_64, 60 GB RAM. Primary model:
**OLMoE-1B-7B-0924-Instruct** (Q4_K_M) unless noted.

---

## Summary

| Scenario | vs llama.cpp | Verdict | Evidence |
|---|---|---|---|
| MoE fits in VRAM | ~same (~49 vs 50 tok/s) | **TIE** | Measured (CPU anchor) |
| MoE bigger than VRAM | ~1.7–2.5× faster | **BETTER** | Projected (`hw_projection`) |
| Long context (KV) | ~26× more context, +0.01 ppl @ 50% | **BETTER** | Measured (`kv_bench`) |
| Multi-turn chat / agents | 2.8–10.6× less prefill | **BETTER** | Measured (`axes_bench`) |
| Many concurrent users | up to ~23× throughput | **BETTER** | Measured (`axes_bench`) |
| VRAM working set | ~2.7× less at ⅜ resident | **BETTER** | Measured (`het_bench`) |
| RAM (Green Compress) | ~45–47% less vs FP32, ~99.8% quality | **BETTER** | Green Compress lab |
| Dense model that fits | modest / lossy | **TIE or worse** | Measured (`dense_bench`) |

The engine wins where work is **sparse** (MoE), **growing** (KV), or **repeated** (chat / batching).
It adds nothing when a small dense model already fits in memory.

### Quality cost

| Lever | Quality impact |
|---|---|
| KV eviction at 50% | +0.01 perplexity |
| KV eviction at 25% | +8% perplexity |
| Expert scheduling | 0 (lossless — never drops a needed expert) |
| Weight compression | Green Compress (~99.8% logit fidelity) |

---

## MoE offload vs llama.cpp

Model: OLMoE-1B-7B, 16 layers, 64 experts, top-8. Output is identical across modes (real routing
replayed; no expert dropped).

### Measured constants

| Quantity | Value | Method |
|---|---:|---|
| llama.cpp decode (Q4, CPU, 8 threads) | 43.6 tok/s | `llama-cpp-python` |
| One expert FFN compute (fp32, 1 token) | 0.109 ms | timed |
| Host memory bandwidth | 73.6 GB/s | timed |
| Cache hit vs miss cost ratio | ~18× | hit ≈ GPU compute; miss ≈ RAM transfer |

### Decode tok/s vs VRAM budget (RAM cold tier)

Full-GPU ceiling (modeled): **654 tok/s**.

| VRAM resident | llama.cpp offload | naive LRU | Green Engine | vs llama.cpp |
|---:|---:|---:|---:|---:|
| 16% | 51.0 | 84.0 | 169.3 | 3.3× |
| 19% | 52.8 | 93.4 | 179.2 | 3.4× |
| 25% | 56.8 | 111.5 | 199.9 | 3.5× |
| 31% | 61.5 | 129.9 | 222.6 | 3.6× |
| 38% | 67.0 | 151.5 | 247.8 | 3.7× |
| 50% | 81.7 | 211.4 | 319.5 | 3.9× |
| 75% | 145.3 | 386.3 | 485.6 | 3.3× |

**Conservative claim:** ~**2×** vs llama.cpp static offload (dynamic cache alone, no prefetch).
**Optimistic:** ~3.3–3.9× with prefetch overlap assumed at 50%.

**SSD cold tier (19% resident):** llama.cpp 52.8 tok/s beats Green Engine 44.3 tok/s — transfer
from NVMe is slower than CPU-computing cold experts. Hybrid per-tier policy is required.

### Expert-set predictor recall (real OLMoE trace)

| Predictor | Recall |
|---|---:|
| Persistence (last token) | 42% |
| Transition / branch-prediction | 43% |
| Hidden-state kNN | **62%** |

---

## Cross-hardware projection

`cargo run --release --bin hw_projection` — ~48 GB Q4 MoE; hit-rate curve from OLMoE trace.

| Device | VRAM | Resident | llama.cpp | Green Engine | Speedup |
|---|---:|---:|---:|---:|---:|
| RTX 3060 | 12 GB | 20% | 191 t/s | 318 | 1.67× |
| RTX 4060 Ti | 16 GB | 26% | 206 t/s | 363 | 1.76× |
| RTX 4090 | 24 GB | 40% | 256 t/s | 553 | 2.16× |
| RTX 5090 | 32 GB | 53% | 327 t/s | 811 | 2.48× |
| AMD 7900 XTX | 24 GB | 40% | 255 t/s | 550 | 2.15× |
| RTX A6000 | 48 GB | 79% | 702 t/s | 1674 | 2.39× |
| A100 80 GB / M2 Max | fits | 100% | — | — | idle |

---

## Fits vs offload (three-way)

| Scenario | vanilla llama.cpp | optimized llama.cpp | Green Engine |
|---|---:|---:|---:|
| OLMoE fits in RAM/VRAM | 49 t/s | 50 t/s | ≈50 t/s |
| Model > VRAM (offload) | slow (static) | slow (static) | **1.7–2.5× faster** |
| Memory footprint | full / static | full / static | **¼ resident + Q8 → up to 16× less** |

llama.cpp vanilla vs optimized (CPU, 4→16 threads, flash-attn): 49.3 → 50.3 tok/s (+2%).

---

## KV, context, and serving

### Context extension (`kv_bench`, OLMoE attention shape)

| Context | Full fp16 KV | Evict 25% | Evict 25% + 2-bit |
|---:|---:|---:|---:|
| 8k | 1.1 GB | 0.27 GB | 0.04 GB |
| 32k | 4.3 GB | 1.07 GB | 0.17 GB |
| 128k | 17.2 GB | 4.29 GB | 0.67 GB |
| 512k | 68.7 GB | 17.2 GB | 2.68 GB |

With 8 GB for KV: full fp16 ~60k tokens; engine policy ~1.5M tokens (~26×).

### Other measured axes

| Axis | Result | Binary |
|---|---|---|
| Prefix-cache reuse | 2.8–10.6× less prefill | `axes_bench` |
| Continuous batching | up to ~23× throughput | `axes_bench` |
| Energy per token | ~3× less (caching) | `energy_bench` |
| Resident memory at 25% cache | 805 MB vs 3221 MB full, ~94% throughput | `portable_bench` |

---

## MCP stack (codehelper)

Separate from MoE decode — powers semantic rerank and orchestration routing.

### green-embed (`ge embed serve --mcp`)

| Metric | Value |
|---|---|
| Backend | ONNX (`onnx-st`) |
| Quality vs PyTorch | 1.0000 cosine |
| Cross-lingual EN↔ES | 0.858 |
| p50 latency | ~9 ms |
| p95 latency | ~23 ms |

Run: `ge embed serve --mcp` then `ge bench mcp`.

### codehelper vs blind file reads

| Task | With MCP | Without MCP |
|---|---|---|
| Find symbol + callers | ~1k tokens, ~180 ms | ~25–40k tokens |

Setup: see README § MCP.

---

## Models tested

| Model | Type | What it tested |
|---|---|---|
| OLMoE-1B-7B | MoE | Expert scheduling, KV, prefetch, hidden-state predictor |
| Qwen2.5-0.5B | Dense | KV quality, dense sparsity limits |
| llama.cpp (OLMoE Q4) | Baseline | Real decode tok/s anchor |

---

## Reproduce

```bash
cargo build --release
cargo test --release

cargo run --release --bin portable_bench   # scheduling hit-rate, bytes/token
cargo run --release --bin runtime_bench    # CPU MoE execution tok/s
cargo run --release --bin kv_bench         # KV eviction + context
cargo run --release --bin axes_bench       # prefix cache + batching
cargo run --release --bin energy_bench     # energy per token
cargo run --release --bin hw_projection    # cross-GPU projection
cargo run --release --bin het_bench        # CPU+GPU split
cargo run --release --bin kernel_bench     # one expert FFN kernel compare

ge bench mcp                               # MCP embed latency (needs embed server)
```

Live llama.cpp baseline:

```bash
ge pull bartowski/OLMoE-1B-7B-0924-Instruct-GGUF
ge run ~/.green/models/<olmoe>.gguf --prompt "test" --ctx 512
```

---

## Caveats

- Offload tok/s above is **projected** from measured hit-rates and constants; wall-clock end-to-end
  on a memory-constrained GPU is the integration milestone (`decode_loop`).
- KV, prefix, and memory numbers are **measured directly**.
- Cross-hardware rows use spec-sheet TFLOPS/bandwidth; routing curves vary by model.

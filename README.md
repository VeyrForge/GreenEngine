# Green Engine

**Run large language models on hardware that should not fit them.**

Green Engine is a memory-smart **local LLM inference** runtime for consumer GPUs and CPUs. It schedules MoE experts, KV cache, and model weights so you get **longer context**, **lower RAM/VRAM use**, and **faster multi-turn chat** — without throwing away quality.

[![Version](https://img.shields.io/badge/version-1.0.0-blue)](crates/ge/Cargo.toml)
[![Rust](https://img.shields.io/badge/rust-stable-orange)](https://rustup.rs/)
[![License: Source-Available](https://img.shields.io/badge/license-Source--Available-orange)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](#quick-start)

**Topics:** `llm` · `local-llm` · `inference` · `moe` · `mixture-of-experts` · `gguf` · `llama-cpp` · `ggml` · `kv-cache` · `rust` · `offline-ai` · `model-serving`

---

## Why Green Engine?

| You want to… | Green Engine helps by… |
|--------------|-------------------------|
| Run a **MoE model bigger than VRAM** | Keeping hot experts in fast memory and paging the rest |
| Chat with **long context** on the same GPU | KV eviction and compression so memory stays bounded |
| Speed up **agents and multi-turn** workflows | Reusing prefix KV instead of recomputing every turn |
| Stay **local and offline** | No cloud API — GGUF models on your machine |

Think of it as an **operating system for inference**: caching, paging, and prefetching for weights and KV, built on portable [ggml](https://github.com/ggml-org/ggml) kernels (CPU, CUDA, Metal, Vulkan, ROCm).

Pair with **[Green Compress](https://github.com/VeyrForge/GreenCompress)** to shrink weight memory (~45% less RAM at ~99.9% quality) before scheduling.

---

## Quick start — the `ge` CLI

One command-line tool to **search**, **download**, **run**, and **benchmark** GGUF models:

```bash
git clone https://github.com/VeyrForge/GreenEngine.git && cd GreenEngine
cargo build --release -p ge
./target/release/ge help
```

```bash
ge models search llama
ge pull bartowski/Llama-3.2-1B-Instruct-GGUF
ge run ~/.green/models/...gguf --prompt "Explain memory bandwidth vs compute for LLMs"
ge bench
ge ui serve    # dashboard at http://127.0.0.1:8780
```

Install the compression companion and wire local embed/chat servers:

```bash
ge install                      # build Green Compress from github.com/VeyrForge/GreenCompress
ge stack setup                  # deps + local MCP profile
ge embed serve                  # embeddings (optional, for codehelper)
ge chat serve                   # OpenAI-compatible local chat
ge compress <args...>           # delegate to greencompress
```

`ge` is dependency-free at compile time — it orchestrates Green Engine and Green Compress without merging their codebases.

---

## When it shines

| Situation | Typical outcome |
|-----------|-----------------|
| MoE model **larger than VRAM** | ~1.7–2.5× faster vs naive offload |
| **Long context** | Much more context in the same memory budget |
| **Multi-turn chat / agents** | Less repeated prefill work |
| Dense model that **already fits** | Use plain llama.cpp — scheduling adds little |

Measured numbers and reproduction steps: **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

---

## Build and test

Requires [Rust stable](https://rustup.rs/). No Python needed for the engine library or benchmarks.

```bash
cargo build --release
cargo test --release
cargo run --release --bin portable_bench
cargo run --release --bin kv_bench
```

Prebuilt binaries ship on **[GitHub Releases](https://github.com/VeyrForge/GreenEngine/releases)** for tagged versions.

---

## Use as a Rust library

```rust
use engine_core::{Config, Engine, Trace, OLMOE_EXPERT_BYTES_FP16};

let trace = Trace::load("crates/engine-core/testdata/expert_trace.bin")?;
let m = Engine::new(Config::full(24, 8), &trace).run(&trace, OLMOE_EXPERT_BYTES_FP16);
println!("hit rate {:.1}%  bytes/token {:.0} MB", m.hit_rate() * 100.0, m.bytes_per_token() / 1e6);
```

Core modules: expert scheduling, KV tiers, continuous batching, weight manifests (Green Compress seam), CPU/GPU hetero execution, energy modeling.

---

## Works with Codehelper (optional)

Index your repo and connect local MCP tools with [Codehelper](https://github.com/VeyrForge/codehelper):

| Service | Command | Env var |
|---------|---------|---------|
| Embeddings | `ge embed serve --mcp` | `CODEHELPER_EMBED_URL=http://127.0.0.1:8766` |
| Chat / routing | `ge chat serve --mcp` | `CODEHELPER_ENRICH_URL=http://127.0.0.1:8767` |

```bash
ge stack setup && ge embed serve --mcp & ge chat serve --mcp &
codehelper init && ge test mcp
```

---

## Project layout

```
crates/ge/           `ge` CLI
crates/engine-core/  scheduling engine + benchmarks
crates/kernels/      native compute (optional CUDA)
runner/              optional Python helpers (embed, chat, UI)
docs/BENCHMARKS.md   performance index
```

---

## License

Free to **run and use** for personal or internal purposes. You may **view** the published source and **submit suggested changes** through the official VeyrForge repository. You may **not** copy, fork, redistribute, create derivative or competing products, or sell this software as your own. See [LICENSE](LICENSE).

Changelog: [CHANGELOG.md](CHANGELOG.md) · Issues and suggestions welcome on GitHub.

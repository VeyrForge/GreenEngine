# Green Engine

**Run large language models on hardware that should not fit them.**

Green Engine is a memory-smart **local LLM inference** runtime for consumer GPUs and CPUs. It schedules MoE experts, KV cache, and model weights so you get **longer context**, **lower RAM/VRAM use**, and **faster multi-turn chat** — without throwing away quality.

[![Version](https://img.shields.io/badge/version-1.0.0-blue)](crates/ge/Cargo.toml)
[![Rust](https://img.shields.io/badge/rust-stable-orange)](https://rustup.rs/)
[![License: Source-Available](https://img.shields.io/badge/license-Source--Available-orange)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](#installation)

---

## Three reasons to use Green Engine

1. **MoE models bigger than VRAM** — Hot experts stay in fast memory; cold ones are paged on demand.
2. **Longer context on the same GPU** — KV eviction and compression keep memory bounded.
3. **One CLI (`ge`)** — Search, pull, run, bench, and serve local GGUF models offline.

---

## Installation

**Prebuilt binaries** (Linux, macOS, Windows): [GitHub Releases](https://github.com/VeyrForge/GreenEngine/releases)

**From source:**

```bash
git clone https://github.com/VeyrForge/GreenEngine.git && cd GreenEngine
cargo build --release -p ge
./target/release/ge help
```

Requires [Rust stable](https://rustup.rs/). Pair with [Green Compress](https://github.com/VeyrForge/GreenCompress) via `ge install` when you need smaller weights.

---

## 30-second example

```bash
ge models search llama
ge pull bartowski/Llama-3.2-1B-Instruct-GGUF
ge run ~/.green/models/.../*.gguf --prompt "Explain KV cache in one paragraph"
```

---

## See it work

No bundled demo video yet — typical `ge` session:

```text
$ ge bench
portable_bench: hit rate 94.2%  bytes/token 12.1 MB

$ ge ui serve
dashboard: http://127.0.0.1:8780

$ ge run model.gguf --prompt "Hello"
61 tokens in 4.1s = 14.8 tok/s
```

Measured numbers and reproduction: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

---

## Supported platforms

| Platform | Notes |
|----------|-------|
| **Linux** | Full support (x86_64, arm64) |
| **macOS** | arm64 + x86_64 release binaries |
| **Windows** | x64 release binaries |

Works with [Codehelper](https://github.com/VeyrForge/codehelper) for local MCP embed/chat (`ge embed serve`, `ge chat serve`).

---

## How it works

Think of Green Engine as an **operating system for inference**: caching, paging, and prefetching for weights and KV, built on portable [ggml](https://github.com/ggml-org/ggml) kernels (CPU, CUDA, Metal, Vulkan, ROCm).

```bash
ge install                      # build Green Compress companion
ge stack setup                  # deps + local MCP profile
ge embed serve                  # embeddings (optional, for codehelper)
ge chat serve                   # OpenAI-compatible local chat
ge compress <args...>           # delegate to greencompress
```

`ge` orchestrates Green Engine and Green Compress without merging their codebases.

---

## Benchmarks

| Situation | Typical outcome |
|-----------|-----------------|
| MoE model **larger than VRAM** | ~1.7–2.5× faster vs naive offload |
| **Long context** | More context in the same memory budget |
| **Multi-turn chat / agents** | Less repeated prefill work |

Full index: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

---

## Documentation

- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — performance index
- [CHANGELOG.md](CHANGELOG.md) — version history
- `ge help` — CLI reference

---

## Limitations

- Dense models that **already fit** in VRAM may see little benefit vs plain llama.cpp.
- Some features require optional [Green Compress](https://github.com/VeyrForge/GreenCompress) for compressed weight tiers.
- GGUF model quality and speed depend on your CPU/GPU and chosen quantization.

---

## Contributing

Issues, benchmark results, and suggested improvements are welcome on [VeyrForge/GreenEngine](https://github.com/VeyrForge/GreenEngine).

Fork the official repository **only** to prepare a pull request back to VeyrForge. See [License and permitted use](#license-and-permitted-use).

---

## Public release history

See [CHANGELOG.md](CHANGELOG.md) and [GitHub Releases](https://github.com/VeyrForge/GreenEngine/releases).

---

## License and permitted use

Green Engine is **source-available** software — not open source.

You may **download, clone, install, inspect, and run** Green Engine for **personal use** or **internal use within your organization**.

You may **fork the official repository solely** for the purpose of preparing and submitting a contribution back to the official VeyrForge repository.

You may **not** redistribute Green Engine, publish modified builds, sell or sublicense it, offer it as a competing hosted inference service, or use its source code to create a competing product without written permission from VeyrForge.

Tutorials may include **short illustrative snippets** from the published source for explanation, provided they do not redistribute the software.

For commercial redistribution, OEM licensing, or other usage not covered above, contact VeyrForge.

This section is a plain-language summary. The binding terms are in [LICENSE](LICENSE).

# Green Engine

**Run large language models on hardware that should not fit them.**

Green Engine is a **local LLM CLI and research runtime** for consumer GPUs and CPUs. The `ge` command searches, pulls, benchmarks, and serves models offline. Scheduling, expert paging, KV experiments, and native `.green` inference are under active development.

[![Version](https://img.shields.io/badge/version-1.1.0-blue)](crates/ge/Cargo.toml)
[![Rust](https://img.shields.io/badge/rust-stable-orange)](https://rustup.rs/)
[![License: Source-Available](https://img.shields.io/badge/license-Source--Available-orange)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](#installation)

---

## What works today

| Capability | Status |
|------------|--------|
| **`ge run` / `ge chat serve` on `.gguf`** | **Working** — launches `llama-cli`, `llama-server`, or `llama_cpp` with ordinary GGUF models (static ggml offload; compatibility mode) |
| **Phase 1 compressed GGUF** | **Working** — `greencompress export-gguf` output runs with `ge chat serve --model file.gguf` |
| **Model search / pull / bench / MCP stack** | **Working** |
| **Dynamic MoE scheduling in generation** | **Not connected** — scheduler, expert cache, paging, and KV experiments exist in `engine-core` but are **not** wired to token generation |
| **Native `.green` packages** | **Phase 2+ (planned/in progress)** — `ge run model.green` validates manifests and returns a clear “runtime not ready” message |
| **Green-compressed inference in `ge run`** | **Not active today** — compression pairs with scheduling in benchmarks; live chat uses exported GGUF via llama.cpp |

---

## Three reasons to use Green Engine

1. **One CLI (`ge`)** — Search, pull, run, bench, compress (via Green Compress), and serve local GGUF models offline.
2. **MCP-friendly local stack** — Embeddings and chat servers for [Codehelper](https://github.com/VeyrForge/codehelper).
3. **Research path for oversized models** — Scheduling, paging, and KV policies validated in benchmarks; native runtime coming in later phases.

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

## 30-second example (GGUF — compatibility mode)

```bash
ge models search llama
ge pull bartowski/Llama-3.2-1B-Instruct-GGUF
ge run ~/.green/models/.../*.gguf --prompt "Explain KV cache in one paragraph"
ge chat serve --model ~/.green/models/.../*.gguf
```

For Green-compressed weights today, export to GGUF first:

```bash
ge install
greencompress export-gguf /path/to/compressed-workdir -o model.gguf
ge chat serve --model model.gguf
```

---

## Roadmap phases

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **Phase 1** | Compressed weights → GGUF fallback for llama.cpp | **Available** via `greencompress export-gguf` |
| **Phase 2+** | Native `.green` packages via Green runtime | **Planned / in progress** (`green-format` crate, `GreenModel` loader stub) |
| **Phase 4** | Paged KV store wired to generation | **Experimental stubs only** (`KvStore` trait; not connected) |

---

## See it work

Typical `ge` session (benchmarks reflect scheduling research, not live chat):

```text
$ ge bench
portable_bench: hit rate 94.2%  bytes/token 12.1 MB

$ ge ui serve
dashboard: http://127.0.0.1:8780

$ ge run model.gguf --prompt "Hello"
compatibility mode — static llama.cpp offload
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

**Today:** `ge` orchestrates llama.cpp (GGUF), Green Compress (weight compression), and optional Python servers (`green_chat.py` remains a fallback, not the primary inference path).

**Experimental (not in generation):** `engine-core` implements MoE expert scheduling, disk paging, hidden-state prefetch, and KV eviction policies — validated in benchmark binaries, not in `ge run` token loops.

```bash
ge install                      # build Green Compress companion
ge stack setup                  # deps + local MCP profile
ge embed serve                  # embeddings (optional, for codehelper)
ge chat serve                   # OpenAI-compatible local chat (GGUF / llama.cpp)
ge compress <args...>           # delegate to greencompress
```

`ge` orchestrates Green Engine and Green Compress without merging their codebases.

---

## Benchmarks

| Situation | Typical outcome (benchmarks) |
|-----------|------------------------------|
| MoE trace under memory pressure | Higher expert hit-rate vs plain LRU |
| **Long context (KV policies)** | More retained attention at same KV budget (simulation) |
| **Compression + scheduling** | Lower bytes/token when manifest reflects compressed experts |

Full index: [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Benchmark results do **not** imply the same speedups in live `ge run` / `ge chat serve` today.

---

## Documentation

- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — performance index
- [CHANGELOG.md](CHANGELOG.md) — version history
- `ge help` — CLI reference

---

## Limitations

- **`ge run` / `ge chat serve` do not use dynamic MoE scheduling or native Green-compressed inference yet.**
- Dense models that **already fit** in VRAM may see little benefit vs plain llama.cpp in compatibility mode.
- GGUF model quality and speed depend on your CPU/GPU and chosen quantization.
- `.green` native packages require Phase 2+ runtime work; use `export-gguf` until then.

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

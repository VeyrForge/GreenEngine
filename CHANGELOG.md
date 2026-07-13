# Changelog

All notable changes to Green Engine follow [Keep a Changelog](https://keepachangelog.com) and
[SemVer](https://semver.org).

**Releases:** **`1.0.0`** · `0.10.0` · `0.9.1` · `0.9.0` · `0.8.2` · `0.8.1` · `0.8.0` ·
`0.7.0` · `0.6.1` · `0.6.0` · `0.5.0` · `0.4.4` · `0.4.3` · `0.4.2` · `0.4.1` · `0.4.0` ·
`0.3.2` · `0.3.1` · `0.3.0` · `0.2.2` · `0.2.1` · `0.2.0` · `0.1.0`

Only **1.0.0** is published on [VeyrForge](https://github.com/VeyrForge/GreenEngine). Earlier
versions are development history documented here.

---

## [1.0.0]

First public release on [VeyrForge](https://github.com/VeyrForge/GreenEngine).

### Added
- **`ge` CLI** and **`engine-core`** at 1.0.0 — MoE scheduling, KV management, benchmarks, MCP
  stack, translation server, and local dashboard (`ge ui serve`).
- **Green Compress integration** — `ge install` / `ge compress` for paired weight compression.
- **`docs/BENCHMARKS.md`** — consolidated performance data and reproduction commands.
- Public README, metadata, and release notes for the VeyrForge distribution.

### Changed
- Public documentation consolidated to `README.md` and `docs/BENCHMARKS.md`.
- MoE trace fixtures committed under `crates/engine-core/testdata/` so tests and benchmarks run
  without Python trace scripts.

### Removed
- `experiments/` trace-capture scripts and internal lab documentation from the public repository.

## [0.10.0]

### Added
- **`ge ui serve`** / **`ge ui install`**: local dashboard for setup, models, compress, bench, and
  chat (http://127.0.0.1:8780).
- GPU-aware chat proxy in the dashboard.
- Project workspace with NDJSON multi-file codegen (auto-continue until WordPress plugin mains exist).
- UI integration tests.

## [0.9.1]

### Changed
- Register-blocked `matvec`: input dimension blocked by 4 for contiguous weight streaming —
  ~1.5× faster CPU decode throughput on memory-bound GEMV.

## [0.9.0]

### Added
- **`green_tiers::allocate_mixed`**: sensitivity-aware mixed-precision allocation — assign more bits
  to sensitive matrices and fewer to robust ones under a fixed RAM budget (GAMMA/HAWQ/CoopQ-style).

## [0.8.2]

### Added
- **`Tensor::Q6G` / `quantize_q6_group`**: group-wise int6 balance tier — nearly lossless at ~2.7×
  compression vs fp32; Pareto-optimal quality/RAM point below int8.

## [0.8.1]

### Added
- **`Tensor::residual_reconstruct`**: usable RVQ tier (Hadamard-NF4 base + int2 residual codebook,
  ~6 bit effective) replacing unusable plain int2 — ~93% fidelity at ~1.6 GB/model in full bench.

### Changed
- GreenTier branded ladder lists only usable precision tiers.

## [0.8.0]

### Added
- **`Tensor::Q2G` / `quantize_q2_group`**: group-wise int2 frontier tier (QuIP#-lite with Hadamard
  incoherence) and branded **GreenTier** precision ladder.
- `full_bench` reports tier sizes and fidelity across the ladder.

## [0.7.0]

### Added
- **`Tensor::hadamard_reconstruct`**: Hadamard rotation quality lever (QuaRot/SpinQuant family).
- **`full_bench`**: head-to-head engine tiers vs Green Compress vs uncompressed, in real KB/MB and
  fidelity %.

## [0.6.1]

### Added
- **NF4 (NormalFloat-4)** tier: non-uniform 4-bit codebook fitted to weight distribution — higher
  fidelity than uniform int4 at the same 0.5 byte/weight.

## [0.6.0]

### Added
- **AWQ** with outlier isolation for weight quantization.
- Live predictive prefetch wired into the runtime prefetch path.
- **`frontier_bench`**: per-expert KB, per-model MB, and expert-output fidelity % vs fp32 in real
  units (not relative L2).

## [0.5.0]

### Added
- **Q4G/Q3G pageable from disk** (`paged.rs`): smallest tiers usable end-to-end without loading full
  fp32 weights into RAM.
- **MSE-optimal clipping** for per-group scales — better quality at the same bit width.

## [0.4.4]

### Added
- **`Tensor::Q3G` / `quantize_q3_group`**: symmetric group-wise int3 (values −4…3), smallest
  dependency-free quality-frontier tier.

## [0.4.3]

### Changed
- GPU int4 expert residency (`expert_cuda.cu`) uses **group-wise scales** — mirrors CPU Q4G quality
  win on the VRAM residency tier.

## [0.4.2]

### Added
- **`Tensor::Q4G` / `quantize_q4_group`**: group-wise int4 (one scale per group of 32–128 elements),
  dependency-free, best quality-per-byte CPU tier.
- Quality sweep benchmarks across tiers.

## [0.4.1]

### Added
- **KV-cache quantization** (`kv.rs` `quantize_kv`).
- Consolidated with/without-Green benchmark report answering compression benefit vs baseline.

## [0.4.0]

### Added
- **Int4 GPU VRAM residency tier** (`GE_GPU_INT4=1` in `expert_cuda.cu`): two signed nibbles per
  byte + per-column scale — ~8× smaller resident expert vs fp32.
- **GPU multi-session** support for concurrent inference sessions.
- Hidden-state predictor integrated into scheduling and prefetch.

## [0.3.2]

### Added
- **`LayerAheadPredictor`** wired into prefetch path; **`predict_bench`** measures recall@k for
  layer-ahead and token-transition prediction on the real OLMoE trace.

## [0.3.1]

### Added
- **Thread-safe stores** (`paged.rs`, `green.rs`) for parallel chat sessions.
- **Async expert prefetch** to hide I/O behind compute.
- **Green-compressed weights on disk** — pageable without full fp32 resident set.

## [0.3.0]

### Added
- **GPU VRAM residency tiers** (`expert_cuda.cu`): fp32 (default), fp16 (`GE_GPU_FP16=1`),
  int8 per-channel (`GE_GPU_INT8=1`) — shrink hot expert cache footprint on GPU.
- **Green Compress integration** through per-expert `WeightManifest` and `ge install` / `ge compress`
  orchestration.

## [0.2.2]

### Added
- **MCP profile**: `ge embed serve --mcp` and `ge chat serve --mcp` (ONNX embed, LRU cache, request
  batching; 1B Q4_K_M chat with 2k context for codehelper enrich/routing).
- **`ge bench mcp`**, **`runner/bench_mcp_stack.py`**, **`runner/start_mcp_stack.sh`**, and
  **`ge stack config`**.
- **`docs/BENCHMARKS.md`** performance index.
- **`--backend`** flag on `green-weights-bench` for CUDA matmul.

### Fixed
- green-embed: SentenceTransformer `backend="onnx"` for cross-lingual rerank quality.
- green-chat: remove unsupported KV cache flags that blocked `llama_cpp.server` startup.

## [0.2.1]

### Fixed
- Cross-platform **`ge` release binaries** on GitHub Releases (linux x86_64/arm64, macOS, Windows).
- Workspace build on clean clones: exclude `green-weights-bench` (path-depends on sibling Green
  Compress checkout).

## [0.2.0]

### Added
- **`ge` CLI**: models search/pull, run, compress, install, and bench.
- **Local MCP stack**: `ge embed serve`, `ge chat serve`, and codehelper integration.
- **`green-weights-bench`**: manifest-driven whole-model benchmark loading Green Compress weights.
- **`ge translate`**: routed translation server on port 8768 (Hy-MT2-7B for 33 languages; GaMS-9B
  for Slovenian auto-routed); OpenAI, Ollama, and Green translate APIs; session metering and pricing.
- **`runner/green_translate.py`**: translation HTTP server.
- Green Compress rebrand (`greencompress` binary) and `ge install` orchestration.
- Removed 14 superseded Python analysis scripts reimplemented in Rust.

## [0.1.0]

First release of the scheduling engine (`engine-core`).

### Added
- **Expert scheduling**: residency cache (LRU/LFU/reuse), transition and hidden-state predictors,
  layer-ahead and speculative-salvage prefetch.
- **KV engine**: eviction (StreamingLLM/H2O/SnapKV/Quest), adaptive per-layer budget, 2-bit model,
  context-extension benchmark.
- **Persistence**: cross-turn and shared-prefix KV reuse.
- **Serving**: continuous batching, chunked prefill, multi-token prediction, disaggregation.
- **Execution**: CPU MoE runtime, Q8 tiered weight store, `ExpertBackend` trait, ggml/CUDA bridge.
- **Heterogeneous** CPU+GPU split and energy / tokens-per-watt model.
- **Green compression seam**: per-expert manifest consumer.
- Hardware detection and backend registry; 11 benchmark binaries; 22 tests.
- CI and GitHub Releases workflow (mirror `release` branch + tagged binary releases).
- Clean release workflow and README de-spam.

[1.0.0]: https://github.com/VeyrForge/GreenEngine/releases/tag/v1.0.0

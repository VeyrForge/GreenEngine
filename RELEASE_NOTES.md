# Green Engine 1.0.0

First public release under the **VeyrForge Source-Available License**.

## Highlights

- **`ge` CLI** — search, pull, run, bench, and serve local GGUF models
- **Memory-smart scheduling** — MoE expert residency, KV paging, prefix reuse (validated in Rust benchmarks)
- **Green Compress integration** — `ge install` / `ge compress` for smaller weights
- **Optional local MCP stack** — embed + chat servers for [Codehelper](https://github.com/VeyrForge/codehelper)
- **Dashboard** — `ge ui serve` at http://127.0.0.1:8780

## Quick start

```bash
git clone https://github.com/VeyrForge/GreenEngine.git && cd GreenEngine
cargo build --release -p ge
./target/release/ge help
```

See [README.md](README.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full details.

## License

Free to run and use; view source and submit suggested changes via GitHub. No fork, redistribution, or competing products without permission. See [LICENSE](LICENSE).

**Full Changelog**: https://github.com/VeyrForge/GreenEngine/commits/v1.0.0

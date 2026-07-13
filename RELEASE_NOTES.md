# Green Engine 1.1.0

Phase 1 honest runtime scope: GGUF llama.cpp compatibility mode, `.green` package validation, and shared manifest types.

## Highlights

- **`green-format` crate** — `green-model` v1 manifest types for `.green` packages from Green Compress `pack-model`
- **`ge run` / `ge chat serve`** — compatibility mode messaging for GGUF; `.green` paths validate with clear Phase 2 errors
- **`GreenModel` loader stub** — parses `pack-model` output; native inference not wired yet
- **Pair with Green Compress `export-gguf`** — llama.cpp fallback for compressed weights

## Quick start

```bash
git clone https://github.com/VeyrForge/GreenEngine.git && cd GreenEngine
cargo build --release -p ge
./target/release/ge help
```

See [README.md](README.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## License

Free to run and use; view source and submit suggested changes via GitHub. See [LICENSE](LICENSE).

**Full Changelog**: https://github.com/VeyrForge/GreenEngine/commits/v1.1.0

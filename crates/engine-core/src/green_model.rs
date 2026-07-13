//! Native `.green` model loader (Phase 2+ runtime seam).
//!
//! Parses Green Compress `pack-model` output and prepares dense/expert stores for the future
//! native inference path. Today this validates packages only; token generation still uses the
//! llama.cpp GGUF fallback (`greencompress export-gguf`).

use std::path::{Path, PathBuf};

use green_format::{open_package, GreenManifest, ModelMetadata, PackageError, TensorRecord};

/// Configuration for opening a `.green` package.
#[derive(Clone, Debug)]
pub struct LoadConfig {
    /// Verify SHA-256 checksums on tensor shards when present.
    pub verify_checksums: bool,
}

impl Default for LoadConfig {
    fn default() -> Self {
        LoadConfig {
            verify_checksums: false,
        }
    }
}

/// Errors from [`GreenModel::open`].
#[derive(Debug)]
pub enum GreenModelError {
    Package(PackageError),
    RuntimeNotReady,
}

impl std::fmt::Display for GreenModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GreenModelError::Package(e) => write!(f, "{e}"),
            GreenModelError::RuntimeNotReady => write!(
                f,
                "Native Green runtime not yet available; use export-gguf for llama.cpp fallback"
            ),
        }
    }
}

impl std::error::Error for GreenModelError {}

impl From<PackageError> for GreenModelError {
    fn from(value: PackageError) -> Self {
        GreenModelError::Package(value)
    }
}

/// Path to dense weights (`dense.gguf` sidecar).
#[derive(Clone, Debug)]
pub struct DenseWeightStore {
    pub path: PathBuf,
}

/// Index of expert tensor shards (Phase 2 paging target).
#[derive(Clone, Debug, Default)]
pub struct GreenExpertStore {
    pub records: Vec<TensorRecord>,
}

impl GreenExpertStore {
    pub fn has_experts(&self) -> bool {
        !self.records.is_empty()
    }
}

/// Tokenizer sidecar (optional until Phase 2 wiring).
#[derive(Clone, Debug, Default)]
pub struct Tokenizer {
    pub path: Option<PathBuf>,
}

/// Handle to a decoded expert weight set (Phase 2 execution stub).
#[derive(Clone, Debug)]
pub struct ExpertHandle {
    pub layer: usize,
    pub expert: u16,
}

/// Expert paging interface for the native Green runtime (not wired to generation yet).
pub trait ExpertProvider {
    fn prefetch(&self, layer: usize, experts: &[u16]);
    fn acquire(&self, layer: usize, expert: u16) -> Result<ExpertHandle, GreenModelError>;
    fn evict(&self, layer: usize, expert: u16);
}

/// Opened `.green` model package.
pub struct GreenModel {
    pub metadata: ModelMetadata,
    pub dense_weights: DenseWeightStore,
    pub experts: GreenExpertStore,
    pub tokenizer: Tokenizer,
    manifest: GreenManifest,
    root: PathBuf,
}

impl GreenModel {
    /// Open a `.green` directory produced by Green Compress `pack-model`.
    ///
    /// Validates `manifest.json`, resolves `metadata.gguf` / `dense.gguf`, and checks expert shard
    /// presence. Returns [`GreenModelError::RuntimeNotReady`] when the package is structurally
    /// valid but the native runtime cannot execute inference yet.
    pub fn open(path: &Path, cfg: &LoadConfig) -> Result<Self, GreenModelError> {
        let pkg = open_package(path, cfg.verify_checksums)?;
        let dense = pkg
            .metadata
            .dense_gguf
            .clone()
            .ok_or_else(|| {
                GreenModelError::Package(PackageError::MissingFile(path.join("dense.gguf")))
            })?;
        let experts: Vec<TensorRecord> = pkg.expert_records().cloned().collect();
        let tokenizer = Tokenizer {
            path: pkg.metadata.tokenizer.clone(),
        };
        let _model = GreenModel {
            metadata: pkg.metadata.clone(),
            dense_weights: DenseWeightStore { path: dense },
            experts: GreenExpertStore { records: experts },
            tokenizer,
            manifest: pkg.manifest.clone(),
            root: pkg.root,
        };
        Err(GreenModelError::RuntimeNotReady)
    }

    pub fn package_root(&self) -> &Path {
        &self.root
    }

    pub fn manifest(&self) -> &GreenManifest {
        &self.manifest
    }
}

/// No-op expert provider until Phase 2 paging is wired.
#[derive(Clone, Copy, Debug, Default)]
pub struct StubExpertProvider;

impl ExpertProvider for StubExpertProvider {
    fn prefetch(&self, _layer: usize, _experts: &[u16]) {}

    fn acquire(&self, _layer: usize, _expert: u16) -> Result<ExpertHandle, GreenModelError> {
        Err(GreenModelError::RuntimeNotReady)
    }

    fn evict(&self, _layer: usize, _expert: u16) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use green_format::manifest::ModelFiles;
    use green_format::schema::{FORMAT_NAME, FORMAT_VERSION};
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn open_reports_runtime_not_ready_for_valid_dense_package() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("metadata.gguf"), b"m").unwrap();
        fs::write(dir.path().join("dense.gguf"), b"d").unwrap();
        let manifest = green_format::GreenManifest {
            format: FORMAT_NAME.into(),
            version: FORMAT_VERSION,
            model: "demo".into(),
            arch: None,
            methods: vec![],
            files: ModelFiles {
                metadata: Some("metadata.gguf".into()),
                dense: Some("dense.gguf".into()),
                tokenizer: None,
            },
            tensors: vec![],
        };
        fs::write(
            dir.path().join("manifest.json"),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
        let res = GreenModel::open(dir.path(), &LoadConfig::default());
        assert!(matches!(res, Err(GreenModelError::RuntimeNotReady)));
    }
}

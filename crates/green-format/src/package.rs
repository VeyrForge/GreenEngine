//! Load a `.green` package directory from disk.

use std::fs;
use std::path::{Path, PathBuf};

use crate::checksum;
use crate::manifest::{GreenManifest, ModelMetadata};
use crate::tensor::TensorRecord;

const MANIFEST_NAME: &str = "manifest.json";

/// Errors while opening a package directory.
#[derive(Debug, PartialEq)]
pub enum PackageError {
    Io(String),
    MissingManifest,
    Json(String),
    InvalidManifest(String),
    MissingFile(PathBuf),
    ChecksumMismatch { path: PathBuf, expected: String },
    ExpertShardsIncomplete { missing: Vec<PathBuf> },
}

impl std::fmt::Display for PackageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PackageError::Io(e) => write!(f, "{e}"),
            PackageError::MissingManifest => write!(f, "missing {MANIFEST_NAME} in package directory"),
            PackageError::Json(e) => write!(f, "manifest JSON: {e}"),
            PackageError::InvalidManifest(e) => write!(f, "invalid manifest: {e}"),
            PackageError::MissingFile(p) => write!(f, "missing package file {}", p.display()),
            PackageError::ChecksumMismatch { path, expected } => {
                write!(f, "checksum mismatch for {} (expected {expected})", path.display())
            }
            PackageError::ExpertShardsIncomplete { missing } => {
                write!(
                    f,
                    "Phase 2 not complete: {} expert shard(s) missing",
                    missing.len()
                )
            }
        }
    }
}

impl std::error::Error for PackageError {}

/// Opened `.green` package with validated manifest and resolved paths.
#[derive(Clone, Debug)]
pub struct GreenPackage {
    pub root: PathBuf,
    pub manifest: GreenManifest,
    pub metadata: ModelMetadata,
}

impl GreenPackage {
    pub fn tensor_path(&self, record: &TensorRecord) -> PathBuf {
        self.root.join(&record.file)
    }

    pub fn expert_records(&self) -> impl Iterator<Item = &TensorRecord> {
        self.manifest.expert_tensors()
    }
}

/// Parse `manifest.json` and verify required sidecar files exist.
pub fn open_package(path: &Path, verify_checksums: bool) -> Result<GreenPackage, PackageError> {
    if !path.is_dir() {
        return Err(PackageError::Io(format!(
            "{} is not a directory",
            path.display()
        )));
    }
    let manifest_path = path.join(MANIFEST_NAME);
    if !manifest_path.is_file() {
        return Err(PackageError::MissingManifest);
    }
    let raw = fs::read_to_string(&manifest_path).map_err(|e| PackageError::Io(e.to_string()))?;
    let manifest: GreenManifest =
        serde_json::from_str(&raw).map_err(|e| PackageError::Json(e.to_string()))?;
    manifest
        .validate()
        .map_err(PackageError::InvalidManifest)?;

    let metadata = manifest.metadata_paths(path);
    for sidecar in [&metadata.metadata_gguf, &metadata.dense_gguf] {
        if let Some(p) = sidecar {
            if !p.is_file() {
                return Err(PackageError::MissingFile(p.clone()));
            }
        }
    }

    let mut missing_experts = Vec::new();
    for rec in manifest.expert_tensors() {
        let shard = path.join(&rec.file);
        if !shard.is_file() {
            missing_experts.push(shard);
            continue;
        }
        if verify_checksums {
            if let Some(expected) = &rec.checksum {
                match checksum::verify_file(&shard, expected) {
                    Ok(true) => {}
                    Ok(false) => {
                        return Err(PackageError::ChecksumMismatch {
                            path: shard,
                            expected: expected.clone(),
                        });
                    }
                    Err(e) => return Err(PackageError::Io(e.to_string())),
                }
            }
        }
    }
    if !missing_experts.is_empty() {
        return Err(PackageError::ExpertShardsIncomplete {
            missing: missing_experts,
        });
    }

    Ok(GreenPackage {
        root: path.to_path_buf(),
        metadata,
        manifest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::ModelFiles;
    use crate::schema::{FORMAT_NAME, FORMAT_VERSION};
    use crate::tensor::{TensorRecord, TensorRole};
    use tempfile::TempDir;

    fn write_manifest(dir: &Path, manifest: &GreenManifest) {
        let p = dir.join(MANIFEST_NAME);
        fs::write(p, serde_json::to_string_pretty(manifest).unwrap()).unwrap();
    }

    #[test]
    fn opens_minimal_dense_package() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("metadata.gguf"), b"meta").unwrap();
        fs::write(dir.path().join("dense.gguf"), b"dense").unwrap();
        let manifest = GreenManifest {
            format: FORMAT_NAME.into(),
            version: FORMAT_VERSION,
            model: "demo".into(),
            arch: Some("llama".into()),
            methods: vec!["green_spqr".into()],
            files: ModelFiles {
                metadata: Some("metadata.gguf".into()),
                dense: Some("dense.gguf".into()),
                tokenizer: None,
            },
            tensors: vec![],
        };
        write_manifest(dir.path(), &manifest);
        let pkg = open_package(dir.path(), false).unwrap();
        assert_eq!(pkg.metadata.model, "demo");
    }

    #[test]
    fn reports_missing_expert_shards() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("metadata.gguf"), b"meta").unwrap();
        fs::write(dir.path().join("dense.gguf"), b"dense").unwrap();
        let manifest = GreenManifest {
            format: FORMAT_NAME.into(),
            version: FORMAT_VERSION,
            model: "moe".into(),
            arch: None,
            methods: vec![],
            files: ModelFiles {
                metadata: Some("metadata.gguf".into()),
                dense: Some("dense.gguf".into()),
                tokenizer: None,
            },
            tensors: vec![TensorRecord {
                name: "blk.0.ffn.expert.0.up".into(),
                role: Some(TensorRole::Expert),
                layer: Some(0),
                expert: Some(0),
                shape: vec![2048, 1024],
                file: "experts/l0_e0.bin".into(),
                offset: 0,
                length: Some(4096),
                checksum: None,
                method: None,
                ggml_type: None,
                source_gguf_type: None,
                green_compression_type: None,
                compressed_size: None,
            }],
        };
        write_manifest(dir.path(), &manifest);
        let err = open_package(dir.path(), false).unwrap_err();
        assert!(matches!(err, PackageError::ExpertShardsIncomplete { .. }));
        assert!(err.to_string().contains("Phase 2 not complete"));
    }
}

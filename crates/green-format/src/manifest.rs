//! `manifest.json` schema for `.green` packages.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::schema::{FORMAT_NAME, FORMAT_VERSION};
use crate::tensor::TensorRecord;

/// Relative paths to bundled GGUF sidecars.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelFiles {
    #[serde(default)]
    pub metadata: Option<String>,
    #[serde(default)]
    pub dense: Option<String>,
    #[serde(default)]
    pub tokenizer: Option<String>,
}

/// Parsed manifest header + tensor index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "ManifestWire")]
pub struct GreenManifest {
    pub format: String,
    pub version: u32,
    pub model: String,
    #[serde(default)]
    pub arch: Option<String>,
    #[serde(default)]
    pub methods: Vec<String>,
    #[serde(default)]
    pub files: ModelFiles,
    #[serde(default)]
    pub tensors: Vec<TensorRecord>,
}

#[derive(Deserialize)]
struct ManifestWire {
    format: String,
    version: u32,
    #[serde(default)]
    model: String,
    #[serde(default, alias = "source_model")]
    source_model: String,
    #[serde(default, alias = "architecture")]
    arch: Option<String>,
    #[serde(default)]
    methods: Vec<String>,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    files: ModelFiles,
    #[serde(default)]
    tensor_files: Vec<String>,
    #[serde(default)]
    tensors: Vec<TensorRecord>,
}

impl From<ManifestWire> for GreenManifest {
    fn from(wire: ManifestWire) -> Self {
        let model = if wire.model.is_empty() {
            wire.source_model
        } else {
            wire.model
        };
        let methods = if wire.methods.is_empty() {
            wire.method.into_iter().collect()
        } else {
            wire.methods
        };
        let mut files = wire.files;
        if files.metadata.is_none() && files.dense.is_none() && !wire.tensor_files.is_empty() {
            for rel in &wire.tensor_files {
                if rel == "metadata.gguf" {
                    files.metadata = Some(rel.clone());
                } else if rel == "dense.gguf" {
                    files.dense = Some(rel.clone());
                }
            }
        }
        GreenManifest {
            format: wire.format,
            version: wire.version,
            model,
            arch: wire.arch,
            methods,
            files,
            tensors: wire.tensors,
        }
    }
}

/// Convenience view over manifest metadata.
#[derive(Clone, Debug)]
pub struct ModelMetadata {
    pub model: String,
    pub arch: Option<String>,
    pub methods: Vec<String>,
    pub metadata_gguf: Option<PathBuf>,
    pub dense_gguf: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
}

impl GreenManifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.format != FORMAT_NAME {
            return Err(format!(
                "unsupported format {:?} (expected {FORMAT_NAME})",
                self.format
            ));
        }
        if self.version != FORMAT_VERSION {
            return Err(format!(
                "unsupported version {} (expected {FORMAT_VERSION})",
                self.version
            ));
        }
        if self.model.is_empty() {
            return Err("manifest model name is empty".into());
        }
        Ok(())
    }

    pub fn metadata_paths(&self, package_root: &Path) -> ModelMetadata {
        ModelMetadata {
            model: self.model.clone(),
            arch: self.arch.clone(),
            methods: self.methods.clone(),
            metadata_gguf: self
                .files
                .metadata
                .as_ref()
                .map(|f| package_root.join(f)),
            dense_gguf: self.files.dense.as_ref().map(|f| package_root.join(f)),
            tokenizer: self.files.tokenizer.as_ref().map(|f| package_root.join(f)),
        }
    }

    pub fn expert_tensors(&self) -> impl Iterator<Item = &TensorRecord> {
        self.tensors.iter().filter(|t| t.is_expert())
    }
}

impl ModelMetadata {
    pub fn from_manifest(manifest: &GreenManifest, package_root: &Path) -> Self {
        manifest.metadata_paths(package_root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorRole;

    #[test]
    fn rejects_wrong_format() {
        let m = GreenManifest {
            format: "gguf".into(),
            version: 1,
            model: "test".into(),
            arch: None,
            methods: vec![],
            files: ModelFiles {
                metadata: None,
                dense: None,
                tokenizer: None,
            },
            tensors: vec![],
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn parses_pack_model_manifest() {
        let raw = include_str!("../tests/fixtures/pack_model_manifest.json");
        let m: GreenManifest = serde_json::from_str(raw).unwrap();
        assert_eq!(m.model, "test-mini.gguf");
        assert_eq!(m.arch.as_deref(), Some("llama"));
        assert_eq!(m.methods, vec!["green_optimal"]);
        assert_eq!(m.files.dense.as_deref(), Some("dense.gguf"));
        assert_eq!(m.files.metadata.as_deref(), Some("metadata.gguf"));
        assert_eq!(m.tensors.len(), 2);
        assert_eq!(m.tensors[0].role, Some(TensorRole::Embedding));
        assert_eq!(m.tensors[1].role, Some(TensorRole::Dense));
    }
}

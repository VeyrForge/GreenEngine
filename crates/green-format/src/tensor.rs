//! Per-tensor records in a Green model manifest.

use serde::{Deserialize, Deserializer, Serialize};

/// Logical role of a tensor within the model graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorRole {
    Dense,
    Expert,
    Embedding,
    Norm,
    Other,
}

fn deserialize_tensor_role<'de, D>(deserializer: D) -> Result<Option<TensorRole>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw: Option<serde_json::Value> = Option::deserialize(deserializer)?;
    Ok(raw.map(|v| match v {
        serde_json::Value::String(s) => classify_tensor_role(&s),
        _ => TensorRole::Other,
    }))
}

fn classify_tensor_role(name: &str) -> TensorRole {
    match name {
        "dense" => TensorRole::Dense,
        "expert" => TensorRole::Expert,
        "embedding" => TensorRole::Embedding,
        "norm" => TensorRole::Norm,
        "other" | "output" => TensorRole::Other,
        s if s.contains("ffn") || s.contains("attn") => TensorRole::Dense,
        _ => TensorRole::Other,
    }
}

/// One tensor entry pointing at a shard inside the package.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TensorRecord {
    pub name: String,
    #[serde(default, deserialize_with = "deserialize_tensor_role")]
    pub role: Option<TensorRole>,
    #[serde(default)]
    pub layer: Option<u32>,
    #[serde(default)]
    pub expert: Option<u16>,
    pub shape: Vec<u32>,
    pub file: String,
    #[serde(default)]
    pub offset: u64,
    #[serde(default)]
    pub length: Option<u64>,
    #[serde(default)]
    pub checksum: Option<String>,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub ggml_type: Option<String>,
    #[serde(default)]
    pub source_gguf_type: Option<String>,
    #[serde(default)]
    pub green_compression_type: Option<String>,
    #[serde(default)]
    pub compressed_size: Option<u64>,
}

impl TensorRecord {
    pub fn is_expert(&self) -> bool {
        matches!(self.role, Some(TensorRole::Expert)) || self.expert.is_some()
    }
}

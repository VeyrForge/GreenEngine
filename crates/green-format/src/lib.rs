//! Shared types for Green `.green` model packages produced by Green Compress `pack-model`.
//!
//! This crate is independent and publishable; Green Engine and (future) Green Compress depend on it.

pub mod checksum;
pub mod manifest;
pub mod package;
pub mod schema;
pub mod tensor;

pub use manifest::{GreenManifest, ModelFiles, ModelMetadata};
pub use package::{open_package, PackageError, GreenPackage};
pub use schema::{FORMAT_NAME, FORMAT_VERSION};
pub use tensor::{TensorRecord, TensorRole};

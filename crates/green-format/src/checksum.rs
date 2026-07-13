//! Optional SHA-256 verification for tensor shards.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

/// Verify a file matches `expected` (`sha256:<hex>` or raw hex).
pub fn verify_file(path: &Path, expected: &str) -> io::Result<bool> {
    let want = parse_checksum(expected).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "invalid checksum format")
    })?;
    let got = sha256_file(path)?;
    Ok(got == want)
}

/// Compute SHA-256 hex digest of `path`.
pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn parse_checksum(s: &str) -> Option<String> {
    let hex = s.strip_prefix("sha256:").unwrap_or(s);
    if hex.len() != 64 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some(hex.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn sha256_roundtrip() {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "green-format").unwrap();
        f.flush().unwrap();
        let digest = sha256_file(f.path()).unwrap();
        assert!(verify_file(f.path(), &format!("sha256:{digest}")).unwrap());
    }
}

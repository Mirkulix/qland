//! Signing keys of team members (Phase 4 spec §4.2): one file per key outside the project,
//! `<key_id>.ed25519` holding the single line `qal-ed25519-secret-v1 <64 hex digits of the seed>`.
//!
//! A secret never leaves this module except as a [`SecretKey`] that signs: it is not printed,
//! logged, stored in the database or put into a diagnostic (messages name only the path and the
//! key id). Seeds and file contents live in memory only inside `Zeroizing`, and `SigningKey`
//! zeroizes itself on drop.

use super::canonical::{hex, hex_into, unhex, unhex_into};
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use qal_core::canon::Hash;
use qal_core::diag::Diagnostic;
use std::path::{Path, PathBuf};
use zeroize::Zeroizing;

/// The first word of a key file.
pub const SECRET_PREFIX: &str = "qal-ed25519-secret-v1";
/// The extension of key files (`.gitignore` has `*.ed25519`).
pub const KEY_EXTENSION: &str = "ed25519";

/// E0920 `key_unusable`.
pub fn unusable(message: impl Into<String>) -> Diagnostic {
    Diagnostic::new("E0920", "key_unusable", "", "", message)
}

/// The key id of a public key: the first 16 hex digits of SHA-256 over its 32 bytes.
pub fn key_id(public: &VerifyingKey) -> String {
    Hash::of(public.as_bytes()).to_hex()[..16].to_string()
}

/// A public key as 64 hex digits.
pub fn public_hex(public: &VerifyingKey) -> String {
    hex(public.as_bytes())
}

/// The public key of 64 hex digits, if they are one.
pub fn parse_public(s: &str) -> Option<VerifyingKey> {
    VerifyingKey::from_bytes(&unhex::<32>(s)?).ok()
}

/// A member's signing key, loaded from its key file. Its `Debug` shows only the key id.
pub struct SecretKey {
    key: SigningKey,
    id: String,
}

impl std::fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecretKey").field("key_id", &self.id).finish_non_exhaustive()
    }
}

impl SecretKey {
    fn from_seed(seed: &Zeroizing<[u8; 32]>) -> SecretKey {
        let key = SigningKey::from_bytes(seed);
        let id = key_id(&key.verifying_key());
        SecretKey { key, id }
    }

    pub fn key_id(&self) -> &str {
        &self.id
    }

    pub fn public(&self) -> VerifyingKey {
        self.key.verifying_key()
    }

    /// The Ed25519 signature of `message` as 128 hex digits.
    pub fn sign_hex(&self, message: &[u8]) -> String {
        hex(&self.key.sign(message).to_bytes())
    }
}

/// 32 bytes from the operating system's random source.
pub fn random32() -> Result<Zeroizing<[u8; 32]>, Diagnostic> {
    let mut b = Zeroizing::new([0u8; 32]);
    getrandom::fill(b.as_mut_slice()).map_err(|e| Diagnostic::store_error(format!("cannot get random bytes from the operating system: {e}")))?;
    Ok(b)
}

/// The directory holding the key files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyDir {
    path: PathBuf,
}

impl KeyDir {
    /// `$QAL_KEY_DIR`, else `$XDG_CONFIG_HOME/qal/keys`, else `~/.config/qal/keys`.
    pub fn from_env() -> Result<KeyDir, Diagnostic> {
        let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
        if let Some(p) = var("QAL_KEY_DIR") {
            return Ok(KeyDir { path: p });
        }
        if let Some(p) = var("XDG_CONFIG_HOME") {
            return Ok(KeyDir { path: p.join("qal").join("keys") });
        }
        match var("HOME") {
            Some(home) => Ok(KeyDir { path: home.join(".config").join("qal").join("keys") }),
            None => Err(unusable("no key directory: set QAL_KEY_DIR (or HOME)")),
        }
    }

    pub fn at(path: impl Into<PathBuf>) -> KeyDir {
        KeyDir { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn file(&self, key_id: &str) -> PathBuf {
        self.path.join(format!("{key_id}.{KEY_EXTENSION}"))
    }

    /// Creates a new key (from the OS random source) and writes its file (0600, never replacing a
    /// file). `project`: the project directory, which must not contain the key directory.
    pub fn generate(&self, project: &Path) -> Result<SecretKey, Diagnostic> {
        self.ready(project, true)?;
        let seed = random32()?;
        let key = SecretKey::from_seed(&seed);
        // Reserved up front: the buffer holding the seed's hex never reallocates.
        let mut line = Zeroizing::new(String::with_capacity(SECRET_PREFIX.len() + 66));
        line.push_str(SECRET_PREFIX);
        line.push(' ');
        hex_into(seed.as_slice(), &mut line);
        line.push('\n');
        write_secret(&self.file(&key.id), line.as_bytes())?;
        Ok(key)
    }

    /// Removes the key file `key_id` (a key generated for a write that was then not committed).
    pub fn discard(&self, key_id: &str) {
        if key_id.len() == 16 && key_id.bytes().all(|b| b.is_ascii_hexdigit()) {
            let _ = std::fs::remove_file(self.file(key_id));
        }
    }

    /// Loads the key `key_id` (the file name must match the key in it).
    pub fn load(&self, project: &Path, key_id: &str) -> Result<SecretKey, Diagnostic> {
        if !(key_id.len() == 16 && key_id.bytes().all(|b| b.is_ascii_hexdigit())) {
            return Err(unusable(format!("`{key_id}` is not a key id")));
        }
        self.ready(project, false)?;
        let path = self.file(key_id);
        let text = read_secret(&path)?;
        let body = text.strip_suffix('\n').unwrap_or(&text);
        let Some((prefix, seed_hex)) = body.split_once(' ') else {
            return Err(unusable(format!("{} is not a QAL key file (key {key_id})", path.display())));
        };
        let mut seed = Zeroizing::new([0u8; 32]);
        if prefix != SECRET_PREFIX || !unhex_into(seed_hex, &mut seed) {
            return Err(unusable(format!("{} is not a QAL key file of format v1 (key {key_id})", path.display())));
        }
        let key = SecretKey::from_seed(&seed);
        if key.id != key_id {
            return Err(unusable(format!("{} holds another key than {key_id}", path.display())).hint("the file was renamed or changed; rekey the member with `qal team rekey <id>`"));
        }
        Ok(key)
    }

    /// The key directory exists (created with 0700 if `create`), is a real directory without group
    /// or other permissions, and is not inside the project (E0926).
    pub fn ready(&self, project: &Path, create: bool) -> Result<PathBuf, Diagnostic> {
        platform::ready(&self.path, project, create)
    }
}

/// `path` resolved through its deepest existing ancestor (which is canonicalized), so a path that
/// does not exist yet can be compared with the project before anything is created.
pub fn resolve(path: &Path) -> std::io::Result<PathBuf> {
    let abs = if path.is_absolute() { path.to_path_buf() } else { std::env::current_dir()?.join(path) };
    let mut rest = Vec::new();
    let mut cur = abs.as_path();
    loop {
        match cur.canonicalize() {
            Ok(c) => return Ok(rest.iter().rev().fold(c, |p: PathBuf, part| p.join(part))),
            Err(e) => {
                let (Some(parent), Some(name)) = (cur.parent(), cur.file_name()) else { return Err(e) };
                rest.push(name.to_os_string());
                cur = parent;
            }
        }
    }
}

/// True if `path` (existing or not) is the project directory or inside it.
pub fn in_project(path: &Path, project: &Path) -> Result<bool, Diagnostic> {
    let canon = resolve(path).map_err(|e| unusable(format!("cannot resolve {}: {e}", path.display())))?;
    let project = project.canonicalize().map_err(|e| Diagnostic::store_error(format!("cannot resolve the project directory {}: {e}", project.display())))?;
    Ok(canon.starts_with(&project))
}

/// The key directory must not be the project or inside it (keys never next to versioned code).
/// Checked before the directory is created, so a refused one is not left behind.
fn outside_project(dir: &Path, project: &Path) -> Result<PathBuf, Diagnostic> {
    let canon = resolve(dir).map_err(|e| unusable(format!("cannot resolve the key directory {}: {e}", dir.display())))?;
    let project = project.canonicalize().map_err(|e| Diagnostic::store_error(format!("cannot resolve the project directory {}: {e}", project.display())))?;
    if canon.starts_with(&project) {
        return Err(Diagnostic::new("E0926", "key_dir_in_project", "", "", format!("the key directory {} is inside the project {}", canon.display(), project.display()))
            .hint("keys never live next to versioned code: set QAL_KEY_DIR to a directory outside the project (default ~/.config/qal/keys)"));
    }
    Ok(canon)
}

#[cfg(unix)]
mod platform {
    use super::{outside_project, unusable};
    use qal_core::diag::Diagnostic;
    use std::io::{Read, Write};
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
    // `create_new` (O_EXCL) never follows a symbolic link; key files are read after a link check.
    use std::path::{Path, PathBuf};
    use zeroize::Zeroizing;

    pub fn ready(dir: &Path, project: &Path, create: bool) -> Result<PathBuf, Diagnostic> {
        outside_project(dir, project)?;
        match std::fs::symlink_metadata(dir) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && create => {
                std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir).map_err(|e| unusable(format!("cannot create the key directory {}: {e}", dir.display())))?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(unusable(format!("there is no key directory {}", dir.display())).hint("set QAL_KEY_DIR to the directory with the team's keys"));
            }
            Err(e) => return Err(unusable(format!("cannot read the key directory {}: {e}", dir.display()))),
            Ok(_) => {}
        }
        let meta = std::fs::symlink_metadata(dir).map_err(|e| unusable(format!("cannot read the key directory {}: {e}", dir.display())))?;
        if meta.file_type().is_symlink() {
            return Err(unusable(format!("the key directory {} is a symbolic link", dir.display())).hint("use the real directory as QAL_KEY_DIR"));
        }
        if !meta.is_dir() {
            return Err(unusable(format!("the key directory {} is not a directory", dir.display())));
        }
        let canon = outside_project(dir, project)?;
        let mode = meta.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            return Err(unusable(format!("the key directory {} has mode {mode:03o}; it must not be accessible by group or others", dir.display())).hint(format!("chmod 700 {}", dir.display())));
        }
        Ok(canon)
    }

    pub fn write_secret(path: &Path, bytes: &[u8]) -> Result<(), Diagnostic> {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .map_err(|e| unusable(format!("cannot create the key file {}: {e}", path.display())))?;
        f.write_all(bytes).and_then(|_| f.sync_all()).map_err(|e| unusable(format!("cannot write the key file {}: {e}", path.display())))
    }

    pub fn read_secret(path: &Path) -> Result<Zeroizing<String>, Diagnostic> {
        let meta = match std::fs::symlink_metadata(path) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(unusable(format!("the key file {} is missing", path.display())).hint("rekey the member with `qal team rekey <id>` (the owner), or set QAL_KEY_DIR to the team's key directory"));
            }
            Err(e) => return Err(unusable(format!("cannot read the key file {}: {e}", path.display()))),
        };
        if meta.file_type().is_symlink() {
            return Err(unusable(format!("the key file {} is a symbolic link", path.display())));
        }
        if !meta.is_file() {
            return Err(unusable(format!("the key file {} is not a file", path.display())));
        }
        let mode = meta.permissions().mode() & 0o777;
        if mode != 0o600 {
            return Err(unusable(format!("the key file {} has mode {mode:03o}, not 600", path.display())).hint(format!("chmod 600 {}", path.display())));
        }
        if meta.len() > 256 {
            return Err(unusable(format!("{} is not a QAL key file (too long)", path.display())));
        }
        let mut f = std::fs::File::open(path).map_err(|e| unusable(format!("cannot read the key file {}: {e}", path.display())))?;
        // Larger than any accepted file: reading never reallocates (and leaves no copy behind).
        let mut text = Zeroizing::new(String::with_capacity(1024));
        f.read_to_string(&mut text).map_err(|e| unusable(format!("cannot read the key file {}: {e}", path.display())))?;
        Ok(text)
    }
}

#[cfg(not(unix))]
mod platform {
    use super::unusable;
    use qal_core::diag::Diagnostic;
    use std::path::{Path, PathBuf};
    use zeroize::Zeroizing;

    const UNSUPPORTED: &str = "team keys need a Unix system (file permissions are checked)";

    pub fn ready(_: &Path, _: &Path, _: bool) -> Result<PathBuf, Diagnostic> {
        Err(unusable(UNSUPPORTED))
    }

    pub fn write_secret(_: &Path, _: &[u8]) -> Result<(), Diagnostic> {
        Err(unusable(UNSUPPORTED))
    }

    pub fn read_secret(_: &Path) -> Result<Zeroizing<String>, Diagnostic> {
        Err(unusable(UNSUPPORTED))
    }
}

use platform::{read_secret, write_secret};

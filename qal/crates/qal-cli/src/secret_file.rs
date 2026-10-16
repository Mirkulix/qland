//! Session token files (Phase 4 spec §4.3): written with mode 0600 by `qal team session`, read by
//! `qal mcp --session-file` only if no group or other user can read them. The token is never
//! printed; in memory it lives in `Zeroizing`.

use qal_core::diag::Diagnostic;
use std::path::Path;
use zeroize::Zeroizing;

fn err(message: String) -> Vec<Diagnostic> {
    vec![Diagnostic::store_error(message)]
}

/// `path` may be written: it is not a symbolic link or a directory.
pub fn check_target(path: &Path) -> Result<(), Vec<Diagnostic>> {
    match std::fs::symlink_metadata(path) {
        Ok(m) if m.file_type().is_symlink() => Err(err(format!("{} is a symbolic link; write the session file to a real path", path.display()))),
        Ok(m) if m.is_dir() => Err(err(format!("{} is a directory", path.display()))),
        _ => Ok(()),
    }
}

/// Writes a secret file with mode 0600 (also when it existed before with other permissions).
#[cfg(unix)]
pub fn write(path: &Path, bytes: &[u8]) -> Result<(), Vec<Diagnostic>> {
    use std::io::Write;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    check_target(path)?;
    let open = || std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(path);
    let mut f = open().map_err(|e| err(format!("cannot write {}: {e}", path.display())))?;
    f.set_permissions(std::fs::Permissions::from_mode(0o600)).map_err(|e| err(format!("cannot set the mode of {}: {e}", path.display())))?;
    f.write_all(bytes).and_then(|_| f.sync_all()).map_err(|e| err(format!("cannot write {}: {e}", path.display())))
}

#[cfg(not(unix))]
pub fn write(_: &Path, _: &[u8]) -> Result<(), Vec<Diagnostic>> {
    Err(err("team sessions need a Unix system (file permissions are checked)".into()))
}

/// Reads a session file: a real file without group or other permissions, holding one token
/// `qs_<64 hex digits>`. The error says why not (never the content).
#[cfg(unix)]
pub fn read_session(path: &Path) -> Result<Zeroizing<String>, String> {
    use std::io::Read;
    use std::os::unix::fs::PermissionsExt;
    let meta = std::fs::symlink_metadata(path).map_err(|e| format!("cannot read the session file {}: {e}", path.display()))?;
    if meta.file_type().is_symlink() || !meta.is_file() {
        return Err(format!("the session file {} is not a regular file", path.display()));
    }
    let mode = meta.permissions().mode() & 0o777;
    if mode & 0o077 != 0 {
        return Err(format!("the session file {} has mode {mode:03o}; it must be readable only by its owner (chmod 600)", path.display()));
    }
    if meta.len() > 256 {
        return Err(format!("{} is not a session file", path.display()));
    }
    let mut text = Zeroizing::new(String::new());
    std::fs::File::open(path).and_then(|mut f| f.read_to_string(&mut text)).map_err(|e| format!("cannot read the session file {}: {e}", path.display()))?;
    let token = Zeroizing::new(text.trim().to_string());
    let well_formed = token.len() == 67 && token.starts_with("qs_") && token[3..].bytes().all(|b| b.is_ascii_hexdigit());
    if !well_formed {
        return Err(format!("{} does not hold a session token", path.display()));
    }
    Ok(token)
}

#[cfg(not(unix))]
pub fn read_session(_: &Path) -> Result<Zeroizing<String>, String> {
    Err("team sessions need a Unix system (file permissions are checked)".into())
}

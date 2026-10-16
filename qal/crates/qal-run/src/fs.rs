//! File access for the `fs` effect through verified directory descriptors (Phase 3 spec §5.3).
//!
//! Every granted directory is opened once, when the permission is parsed, as a descriptor
//! (`O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC`). A guest path is resolved **lexically** against the
//! granted directories (`.` and `..` are applied to the text; a path that leaves every granted
//! directory, a NUL byte, or a hidden component under `no_hidden` is refused uniformly) and then
//! opened component by component with `openat(dirfd, component, O_DIRECTORY | O_NOFOLLOW)`, the
//! last one with `O_NOFOLLOW | O_NONBLOCK`. A symbolic link anywhere below the granted directory
//! makes `openat` fail, so nothing outside can be reached or created, whatever another process
//! swaps in between two steps; FIFOs and devices never block and are refused after `fstat`.

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

/// A granted directory: its canonical path, the absolute path it was granted as (both are accepted
/// in guest paths), and the descriptor every access starts from.
#[derive(Clone)]
pub struct FsRoot {
    pub path: PathBuf,
    alias: PathBuf,
    #[cfg(unix)]
    fd: Arc<std::os::fd::OwnedFd>,
    #[cfg(not(unix))]
    _fd: Arc<()>,
}

impl std::fmt::Debug for FsRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FsRoot({})", self.path.display())
    }
}

impl PartialEq for FsRoot {
    fn eq(&self, other: &FsRoot) -> bool {
        self.path == other.path
    }
}

/// Lexically normalizes an absolute path (`.` dropped, `..` removes the previous component).
fn normalize(p: &Path) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::Prefix(_) | Component::RootDir => out.push(c.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    return None;
                }
            }
            Component::Normal(n) => out.push(n),
        }
    }
    Some(out)
}

impl FsRoot {
    /// Opens `dir` (relative paths against the working directory) as a granted directory.
    pub fn open(dir: &Path) -> Result<FsRoot, String> {
        let shown = dir.display();
        let canonical = dir.canonicalize().map_err(|e| format!("fs:{shown}: {e}"))?;
        if !canonical.is_dir() {
            return Err(format!("fs:{shown}: not a directory"));
        }
        let alias = std::path::absolute(dir).ok().and_then(|a| normalize(&a)).unwrap_or_else(|| canonical.clone());
        #[cfg(unix)]
        {
            use rustix::fs::{Mode, OFlags};
            let fd = rustix::fs::open(&canonical, OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC, Mode::empty()).map_err(|e| format!("fs:{shown}: {}", std::io::Error::from(e)))?;
            Ok(FsRoot { path: canonical, alias, fd: Arc::new(fd) })
        }
        #[cfg(not(unix))]
        {
            let _ = alias;
            Err(format!("fs:{shown}: the `fs` effect is not supported on this platform"))
        }
    }

    /// The components of `abs` below this root, if it lies inside (by its canonical path or the
    /// path it was granted as).
    fn below<'p>(&self, abs: &'p Path) -> Option<&'p Path> {
        abs.strip_prefix(&self.path).ok().or_else(|| abs.strip_prefix(&self.alias).ok())
    }
}

/// How a file is opened.
#[derive(Clone, Copy, PartialEq)]
pub enum Access {
    Read,
    /// Created if missing (mode 0o644); truncated only after it is known to be a regular file.
    Write,
}

/// Uniform refusal: identical for paths that exist and paths that do not.
pub fn denied(path: &str) -> String {
    format!("permission denied: `{path}` is outside the allowed directories")
}

/// Resolves the guest `path` against `base` (for relative paths; absolute paths stay) and finds
/// the granted root it lies in, with the components below that root.
pub fn locate<'r>(roots: &'r [FsRoot], base: &Path, path: &str, no_hidden: bool) -> Result<(&'r FsRoot, Vec<String>), String> {
    if path.contains('\0') || path.is_empty() {
        return Err(denied(path));
    }
    let joined = base.join(path);
    let abs = normalize(&joined).ok_or_else(|| denied(path))?;
    // The innermost root wins (a granted directory inside another one).
    let mut best: Option<(&FsRoot, Vec<String>)> = None;
    for r in roots {
        let Some(rel) = r.below(&abs) else { continue };
        let comps: Option<Vec<String>> = rel.components().map(|c| c.as_os_str().to_str().map(str::to_string)).collect();
        let Some(comps) = comps else { continue };
        if best.as_ref().is_none_or(|(_, b)| comps.len() < b.len()) {
            best = Some((r, comps));
        }
    }
    let (root, comps) = best.ok_or_else(|| denied(path))?;
    if no_hidden && comps.iter().any(|c| c.starts_with('.')) {
        return Err(denied(path));
    }
    Ok((root, comps))
}

/// Opens the guest `path` inside the granted `roots` without following any symbolic link below
/// the root and without blocking; the result is a regular file (for `Write`: not yet truncated).
#[cfg(unix)]
pub fn open(roots: &[FsRoot], base: &Path, path: &str, no_hidden: bool, access: Access) -> Result<std::fs::File, String> {
    use rustix::fs::{FileType, Mode, OFlags};
    use std::os::fd::{AsFd, OwnedFd};
    let (root, comps) = locate(roots, base, path, no_hidden)?;
    let Some((last, dirs)) = comps.split_last() else {
        return Err(format!("{path}: not a regular file"));
    };
    let io = |e: rustix::io::Errno| -> String {
        // A symbolic link on the way (ELOOP from O_NOFOLLOW, ENOTDIR from O_DIRECTORY on a link)
        // is refused like a path outside.
        if e == rustix::io::Errno::LOOP || e == rustix::io::Errno::NOTDIR || e == rustix::io::Errno::MLINK {
            denied(path)
        } else {
            format!("{path}: {}", std::io::Error::from(e))
        }
    };
    let dir_flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let mut current: Option<OwnedFd> = None;
    for d in dirs {
        let next = match &current {
            None => rustix::fs::openat(root.fd.as_fd(), d.as_str(), dir_flags, Mode::empty()),
            Some(fd) => rustix::fs::openat(fd.as_fd(), d.as_str(), dir_flags, Mode::empty()),
        }
        .map_err(io)?;
        current = Some(next);
    }
    let flags = OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC
        | match access {
            Access::Read => OFlags::RDONLY,
            Access::Write => OFlags::WRONLY | OFlags::CREATE,
        };
    let mode = Mode::from_raw_mode(0o644);
    let fd = match &current {
        None => rustix::fs::openat(root.fd.as_fd(), last.as_str(), flags, mode),
        Some(dir) => rustix::fs::openat(dir.as_fd(), last.as_str(), flags, mode),
    }
    .map_err(io)?;
    let st = rustix::fs::fstat(&fd).map_err(io)?;
    if FileType::from_raw_mode(st.st_mode) != FileType::RegularFile {
        return Err(format!("{path}: not a regular file"));
    }
    // A hard link may name a file outside the granted directory; links cannot be told apart, so
    // files with several names are refused.
    if st.st_nlink > 1 {
        return Err(denied(path));
    }
    if access == Access::Write {
        rustix::fs::ftruncate(&fd, 0).map_err(io)?;
    }
    Ok(std::fs::File::from(fd))
}

#[cfg(not(unix))]
pub fn open(_: &[FsRoot], _: &Path, path: &str, _: bool, _: Access) -> Result<std::fs::File, String> {
    Err(format!("{path}: the `fs` effect is not supported on this platform"))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("qal-fs-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("data/sub")).unwrap();
        d
    }

    fn read(roots: &[FsRoot], base: &Path, p: &str) -> Result<String, String> {
        let mut s = String::new();
        open(roots, base, p, false, Access::Read)?.read_to_string(&mut s).unwrap();
        Ok(s)
    }

    #[test]
    fn paths_resolve_lexically_inside_the_granted_directories() {
        let d = dir("lex");
        std::fs::write(d.join("data/sub/a.txt"), "A").unwrap();
        std::fs::write(d.join("secret.txt"), "S").unwrap();
        let roots = vec![FsRoot::open(&d.join("data")).unwrap()];
        let base = d.join("data");
        assert_eq!(read(&roots, &base, "sub/a.txt").unwrap(), "A");
        assert_eq!(read(&roots, &base, "./sub/../sub/a.txt").unwrap(), "A");
        assert_eq!(read(&roots, &base, &d.join("data/sub/a.txt").to_string_lossy()).unwrap(), "A", "absolute path as granted");
        assert_eq!(read(&roots, &base, &d.canonicalize().unwrap().join("data/sub/a.txt").to_string_lossy()).unwrap(), "A", "canonical path");
        for outside in ["../secret.txt", "sub/../../secret.txt", "/etc/hosts", "/nonexistent/../etc/hosts", "", "a\0b"] {
            assert_eq!(read(&roots, &base, outside).unwrap_err(), denied(outside), "{outside:?}");
        }
        assert!(read(&roots, &base, "sub/missing.txt").unwrap_err().starts_with("sub/missing.txt: No such file"));
        assert_eq!(read(&roots, &base, "sub").unwrap_err(), "sub: not a regular file");
        assert_eq!(read(&roots, &base, ".").unwrap_err(), ".: not a regular file");
        assert_eq!(locate(&roots, &base, "sub/.env", true).unwrap_err(), denied("sub/.env"), "hidden components are refused under no_hidden");
        assert_eq!(locate(&roots, &base, "sub/.env", false).unwrap().1, vec!["sub".to_string(), ".env".to_string()]);
    }

    #[test]
    fn symlinks_anywhere_below_the_root_are_refused() {
        let d = dir("links");
        std::fs::write(d.join("secret.txt"), "S").unwrap();
        std::os::unix::fs::symlink(&d, d.join("data/up")).unwrap();
        std::os::unix::fs::symlink(d.join("secret.txt"), d.join("data/file-link")).unwrap();
        std::os::unix::fs::symlink(d.join("nowhere.txt"), d.join("data/dangling")).unwrap();
        let roots = vec![FsRoot::open(&d.join("data")).unwrap()];
        let base = d.join("data");
        for p in ["up/secret.txt", "file-link", "dangling"] {
            assert_eq!(read(&roots, &base, p).unwrap_err(), denied(p), "{p}");
            assert_eq!(open(&roots, &base, p, false, Access::Write).unwrap_err(), denied(p), "{p}");
        }
        assert!(!d.join("nowhere.txt").exists(), "writing through a dangling link creates nothing");
        assert_eq!(std::fs::read_to_string(d.join("secret.txt")).unwrap(), "S");
    }

    #[test]
    fn swapping_a_directory_for_a_symlink_creates_nothing_outside() {
        // The race of Phase 1 (`check_path`, then a swap, then `open`) cannot happen: each component
        // is opened relative to the verified descriptor of its parent.
        let d = dir("swap");
        let outside = d.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        let roots = vec![FsRoot::open(&d.join("data")).unwrap()];
        let base = d.join("data");
        // Step 1: the path is fine.
        open(&roots, &base, "sub/new.txt", false, Access::Write).unwrap().write_all(b"x").unwrap();
        // Step 2: `sub` becomes a link to a directory outside.
        std::fs::remove_dir_all(d.join("data/sub")).unwrap();
        std::os::unix::fs::symlink(&outside, d.join("data/sub")).unwrap();
        assert_eq!(open(&roots, &base, "sub/new2.txt", false, Access::Write).unwrap_err(), denied("sub/new2.txt"));
        assert_eq!(std::fs::read_dir(&outside).unwrap().count(), 0, "no file was created outside");
        // The granted directory itself is held open: replacing it does not redirect the descriptor.
        std::fs::rename(d.join("data"), d.join("moved")).unwrap();
        std::os::unix::fs::symlink(&outside, d.join("data")).unwrap();
        open(&roots, &base, "kept.txt", false, Access::Write).unwrap();
        assert!(d.join("moved/kept.txt").exists() && !outside.join("kept.txt").exists());
    }

    #[test]
    fn hard_links_to_files_outside_are_refused() {
        let d = dir("hard");
        std::fs::write(d.join("outside.txt"), "precious").unwrap();
        std::fs::hard_link(d.join("outside.txt"), d.join("data/link.txt")).unwrap();
        let roots = vec![FsRoot::open(&d.join("data")).unwrap()];
        assert_eq!(open(&roots, &d.join("data"), "link.txt", false, Access::Write).unwrap_err(), denied("link.txt"));
        assert_eq!(read(&roots, &d.join("data"), "link.txt").unwrap_err(), denied("link.txt"));
        assert_eq!(std::fs::read_to_string(d.join("outside.txt")).unwrap(), "precious");
    }

    #[test]
    fn writing_truncates_only_regular_files() {
        let d = dir("trunc");
        std::fs::write(d.join("data/a.txt"), "long old content").unwrap();
        let roots = vec![FsRoot::open(&d.join("data")).unwrap()];
        let mut f = open(&roots, &d.join("data"), "a.txt", false, Access::Write).unwrap();
        f.write_all(b"new").unwrap();
        assert_eq!(std::fs::read_to_string(d.join("data/a.txt")).unwrap(), "new");
        assert!(std::process::Command::new("mkfifo").arg(d.join("data/pipe")).status().unwrap().success());
        let t = std::time::Instant::now();
        assert!(open(&roots, &d.join("data"), "pipe", false, Access::Read).unwrap_err().ends_with("not a regular file"));
        assert!(open(&roots, &d.join("data"), "pipe", false, Access::Write).unwrap_err().starts_with("pipe: "));
        assert!(t.elapsed() < std::time::Duration::from_secs(2));
        assert!(FsRoot::open(&d.join("data/a.txt")).unwrap_err().contains("not a directory"));
        assert!(FsRoot::open(&d.join("nope")).is_err());
    }
}

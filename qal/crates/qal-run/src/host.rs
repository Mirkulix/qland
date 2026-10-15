//! Host functions (`qal_host.*`) and the permission model: only granted effect groups are linked
//! (ungranted ones are stubs that fail with `permission_denied`), and `fs` only reaches files
//! inside granted directories.

use crate::value::read_str;
use qal_wasm::{host, FAIL_OOM, HOST_MODULE, OOM_SCRATCH};
use std::fs::File;
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use wasmtime::{Caller, Extern, FuncType, Linker, Memory, StoreLimits, ValType};

/// Failure kind recorded by the stub of an ungranted effect function; `func` indexes `host::EFFECT_FNS`.
pub const FAIL_DENIED: i32 = -1;

/// Effect permissions for one execution, e.g. parsed from `--allow io,fs:./data`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Permissions {
    pub io: bool,
    pub time: bool,
    pub rand: bool,
    /// Canonical directories `fs` may read and write below. Empty = no `fs`.
    pub fs: Vec<PathBuf>,
}

impl Permissions {
    /// Parses a comma-separated list: `io`, `time`, `rand`, `fs:<dir>` (repeatable). Directories must exist.
    pub fn parse(spec: &str) -> Result<Permissions, String> {
        let mut p = Permissions::default();
        for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match item {
                "io" => p.io = true,
                "time" => p.time = true,
                "rand" => p.rand = true,
                "fs" => return Err("`fs` needs a directory: `fs:<dir>`, e.g. `fs:./data`".into()),
                _ => match item.strip_prefix("fs:") {
                    Some(dir) => {
                        require_nofollow(O_NOFOLLOW)?;
                        let canonical = Path::new(dir).canonicalize().map_err(|e| format!("fs:{dir}: {e}"))?;
                        if !canonical.is_dir() {
                            return Err(format!("fs:{dir}: not a directory"));
                        }
                        p.fs.push(canonical);
                    }
                    None => return Err(format!("unknown effect `{item}` (known: io, fs:<dir>, time, rand)")),
                },
            }
        }
        Ok(p)
    }

    pub fn allows(&self, effect: &str) -> bool {
        match effect {
            "io" => self.io,
            "time" => self.time,
            "rand" => self.rand,
            "fs" => !self.fs.is_empty(),
            _ => false,
        }
    }

    fn inside(&self, p: &Path) -> bool {
        self.fs.iter().any(|root| p.starts_with(root))
    }

    /// Uniform refusal: identical for paths that exist and paths that do not, so file existence
    /// outside the granted directories cannot be probed.
    fn denied(path: &str) -> String {
        format!("permission denied: `{path}` is outside the allowed directories")
    }

    /// A path that could not be resolved: the real error is only shown if the nearest existing
    /// ancestor lies inside a granted directory, otherwise it is the uniform refusal.
    fn unresolved(&self, p: &Path, path: &str, err: std::io::Error) -> String {
        for anc in p.ancestors().skip(1) {
            let anc = if anc.as_os_str().is_empty() { Path::new(".") } else { anc };
            if let Ok(c) = anc.canonicalize() {
                return if self.inside(&c) { format!("{path}: {err}") } else { Self::denied(path) };
            }
        }
        Self::denied(path)
    }

    /// Resolves `path` (relative to the working directory) and checks it lies inside a granted directory.
    /// For writing, the file itself may not exist yet, but its directory must.
    pub fn check_path(&self, path: &str, for_write: bool) -> Result<PathBuf, String> {
        let p = Path::new(path);
        let resolved = match p.canonicalize() {
            Ok(c) => c,
            Err(e) if !for_write => return Err(self.unresolved(p, path, e)),
            Err(e) => {
                let parent = match p.parent() {
                    Some(d) if !d.as_os_str().is_empty() => d,
                    _ => Path::new("."),
                };
                let Some(name) = p.file_name() else { return Err(self.unresolved(p, path, e)) };
                let full = match parent.canonicalize() {
                    Ok(d) => d.join(name),
                    Err(e) => return Err(self.unresolved(p, path, e)),
                };
                if self.inside(&full) && std::fs::symlink_metadata(p).is_ok() {
                    // A dangling symlink "does not exist" but writing would follow it, possibly outside.
                    return Err(format!("permission denied: `{path}` is a dangling symbolic link"));
                }
                full
            }
        };
        if self.inside(&resolved) {
            Ok(resolved)
        } else {
            Err(Self::denied(path))
        }
    }

    /// After opening, re-resolves `resolved` and requires that it is still inside a granted directory
    /// and is the very file that was opened (guards against links swapped in after `check_path`).
    pub fn verify_opened(&self, file: &File, resolved: &Path) -> Result<(), String> {
        let changed = || format!("permission denied: `{}` changed while being opened", resolved.display());
        let now = resolved.canonicalize().map_err(|_| changed())?;
        if !self.inside(&now) {
            return Err(changed());
        }
        let (a, b) = (file.metadata().map_err(|_| changed())?, std::fs::metadata(&now).map_err(|_| changed())?);
        if same_file(&a, &b) {
            Ok(())
        } else {
            Err(changed())
        }
    }

    /// Checks, opens without following a final symlink, and verifies the opened file.
    fn open(&self, path: &str, for_write: bool) -> Result<File, String> {
        let full = self.check_path(path, for_write)?;
        let file = open_nofollow(&full, for_write).map_err(|e| format!("{path}: {e}"))?;
        self.verify_opened(&file, &full)?;
        Ok(file)
    }
}

#[cfg(unix)]
fn same_file(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    a.dev() == b.dev() && a.ino() == b.ino()
}

#[cfg(not(unix))]
fn same_file(_: &std::fs::Metadata, _: &std::fs::Metadata) -> bool {
    true
}

/// `O_NOFOLLOW` where its value is known; `fs` is refused elsewhere (and on non-Unix targets,
/// where `same_file` cannot verify the opened file) instead of silently opening less safely.
const O_NOFOLLOW: Option<i32> = if cfg!(all(
    unix,
    any(target_os = "macos", target_os = "ios", target_os = "freebsd", target_os = "netbsd", target_os = "openbsd", target_os = "dragonfly")
)) {
    Some(0x0100)
} else if cfg!(all(
    any(target_os = "linux", target_os = "android"),
    any(target_arch = "aarch64", target_arch = "arm", target_arch = "powerpc", target_arch = "powerpc64", target_arch = "m68k")
)) {
    Some(0o100000)
} else if cfg!(all(
    any(target_os = "linux", target_os = "android"),
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "riscv32",
        target_arch = "riscv64",
        target_arch = "loongarch64",
        target_arch = "s390x",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "sparc",
        target_arch = "sparc64"
    )
)) {
    Some(0o400000)
} else {
    None
};

fn require_nofollow(flag: Option<i32>) -> Result<i32, String> {
    flag.ok_or_else(|| "the `fs` effect is not supported on this platform: files cannot be opened without following symbolic links".into())
}

/// Opens for reading, or for writing (created if missing, not truncated: the caller truncates once verified).
///
/// Known limitation: only the final path component is opened without following links, so a
/// co-located process that swaps a directory of the path for a symlink between `check_path` and
/// this call can make a write create an EMPTY file outside the granted directories (`verify_opened`
/// then refuses to write to it). The proper fix, opening with `openat` relative to a verified
/// parent directory descriptor, is planned for Phase 3.
fn open_nofollow(path: &Path, write: bool) -> std::io::Result<File> {
    let flag = require_nofollow(O_NOFOLLOW).map_err(std::io::Error::other)?;
    let mut o = std::fs::OpenOptions::new();
    if write {
        o.write(true).create(true);
    } else {
        o.read(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.custom_flags(flag);
    }
    #[cfg(not(unix))]
    let _ = flag;
    o.open(path)
}

fn read_text(file: File, path: &str, limit: usize) -> Result<Result<String, String>, usize> {
    let len = file.metadata().map(|m| m.len()).unwrap_or(0) as usize;
    if len > limit {
        return Err(len);
    }
    let mut bytes = Vec::new();
    if let Err(e) = file.take(limit as u64 + 1).read_to_end(&mut bytes) {
        return Ok(Err(format!("{path}: {e}")));
    }
    if bytes.len() > limit {
        return Err(bytes.len());
    }
    Ok(String::from_utf8(bytes).map_err(|_| format!("{path}: not valid UTF-8")))
}

/// A failure reported by the guest through `qal_host.fail`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Failure {
    pub kind: i32,
    pub func: i32,
    pub site: i32,
    pub vals: i32,
}

pub struct HostState {
    pub failure: Option<Failure>,
    pub stdout: Vec<u8>,
    pub stdin: Box<dyn BufRead + Send>,
    pub perms: Permissions,
    pub rng: u64,
    pub limits: StoreLimits,
    /// Memory limit in bytes; also caps captured stdout and host-side reads.
    pub limit: usize,
}

fn memory(c: &mut Caller<'_, HostState>) -> Result<Memory, wasmtime::Error> {
    match c.get_export("memory") {
        Some(Extern::Memory(m)) => Ok(m),
        _ => Err(wasmtime::Error::msg("module has no memory export")),
    }
}

fn guest_str(c: &mut Caller<'_, HostState>, ptr: i32) -> Result<String, wasmtime::Error> {
    let mem = memory(c)?;
    read_str(mem.data(&c), ptr as u32).ok_or_else(|| wasmtime::Error::msg("invalid string pointer"))
}

/// Reports a host-side allocation over the memory limit exactly like a guest allocation failure.
fn host_oom(c: &mut Caller<'_, HostState>, requested: usize) -> wasmtime::Error {
    if let Ok(mem) = memory(c) {
        let _ = mem.write(&mut *c, OOM_SCRATCH as usize, &(requested as i64).to_le_bytes());
    }
    c.data_mut().failure = Some(Failure { kind: FAIL_OOM, func: 0, site: 0, vals: 0 });
    wasmtime::Error::msg("out of memory")
}

/// Allocates a Str in guest memory via the exported allocator (which may fail with out_of_memory).
pub fn new_str(c: &mut Caller<'_, HostState>, s: &str) -> Result<i32, wasmtime::Error> {
    let alloc = c.get_export("qal_alloc").and_then(Extern::into_func).ok_or_else(|| wasmtime::Error::msg("no qal_alloc"))?;
    let p = alloc.typed::<i64, i32>(&c)?.call(&mut *c, 4 + s.len() as i64)?;
    let mem = memory(c)?;
    mem.write(&mut *c, p as usize, &(s.len() as u32).to_le_bytes())?;
    mem.write(&mut *c, p as usize + 4, s.as_bytes())?;
    Ok(p)
}

fn val_type(t: &host::ValType) -> Result<ValType, wasmtime::Error> {
    match t {
        host::ValType::I32 => Ok(ValType::I32),
        host::ValType::I64 => Ok(ValType::I64),
        host::ValType::F64 => Ok(ValType::F64),
        other => Err(wasmtime::Error::msg(format!("host signatures use no {other:?}"))),
    }
}

/// splitmix64
fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Defines `fail`, the pure helpers, the functions of every granted effect, and stubs for the
/// ungranted ones: a module may import them (a fn value of an effectful fn makes it reachable)
/// without ever calling them, so it must still instantiate.
pub fn link(linker: &mut Linker<HostState>, perms: &Permissions) -> Result<(), wasmtime::Error> {
    for (k, name) in host::EFFECT_FNS.iter().enumerate() {
        if perms.allows(host::effect(name).unwrap_or("")) {
            continue;
        }
        let (params, results) = host::signature(name).ok_or_else(|| wasmtime::Error::msg(format!("unknown host function `{name}`")))?;
        let params = params.iter().map(val_type).collect::<Result<Vec<_>, _>>()?;
        let results = results.iter().map(val_type).collect::<Result<Vec<_>, _>>()?;
        let ty = FuncType::new(linker.engine(), params, results);
        linker.func_new(HOST_MODULE, name, ty, move |mut c, _, _| {
            c.data_mut().failure = Some(Failure { kind: FAIL_DENIED, func: k as i32, site: 0, vals: 0 });
            Err(wasmtime::Error::msg("permission denied"))
        })?;
    }
    linker.func_wrap(HOST_MODULE, host::FAIL, |mut c: Caller<'_, HostState>, kind: i32, func: i32, site: i32, vals: i32| {
        c.data_mut().failure = Some(Failure { kind, func, site, vals });
        Err::<(), _>(wasmtime::Error::msg("qal failure"))
    })?;
    linker.func_wrap(HOST_MODULE, host::FLOAT_TO_STR, |mut c: Caller<'_, HostState>, x: f64| new_str(&mut c, &format!("{x:?}")))?;
    linker.func_wrap(HOST_MODULE, host::STR_TO_FLOAT, |mut c: Caller<'_, HostState>, p: i32| {
        Ok(match guest_str(&mut c, p)?.parse::<f64>() {
            Ok(x) => (1, x),
            Err(_) => (0, 0.0),
        })
    })?;
    if perms.io {
        linker.func_wrap(HOST_MODULE, host::PRINT, |mut c: Caller<'_, HostState>, p: i32| {
            let s = guest_str(&mut c, p)?;
            let limit = c.data().limit;
            let out = &mut c.data_mut().stdout;
            let mut line = s.into_bytes();
            line.push(b'\n');
            let requested = out.len() + line.len();
            if requested > limit {
                out.extend_from_slice(&line[..limit.saturating_sub(out.len())]);
                return Err(host_oom(&mut c, requested));
            }
            out.extend_from_slice(&line);
            Ok(())
        })?;
        linker.func_wrap(HOST_MODULE, host::READ_LINE, |mut c: Caller<'_, HostState>| {
            let mut buf = Vec::new();
            let limit = c.data().limit;
            let n = c.data_mut().stdin.by_ref().take(limit as u64 + 1).read_until(b'\n', &mut buf).map_err(wasmtime::Error::msg)?;
            if n == 0 {
                return Ok(-1);
            }
            if n > limit {
                return Err(host_oom(&mut c, n));
            }
            if buf.ends_with(b"\n") {
                buf.pop();
                if buf.ends_with(b"\r") {
                    buf.pop();
                }
            }
            new_str(&mut c, &String::from_utf8_lossy(&buf))
        })?;
    }
    if !perms.fs.is_empty() {
        linker.func_wrap(HOST_MODULE, host::READ_FILE, |mut c: Caller<'_, HostState>, p: i32| {
            let path = guest_str(&mut c, p)?;
            let limit = c.data().limit;
            let result = match c.data().perms.open(&path, false) {
                Ok(file) => match read_text(file, &path, limit) {
                    Ok(r) => r,
                    Err(size) => return Err(host_oom(&mut c, size)),
                },
                Err(e) => Err(e),
            };
            let (ok, text) = match result {
                Ok(s) => (1, s),
                Err(e) => (0, e),
            };
            Ok((ok, new_str(&mut c, &text)?))
        })?;
        linker.func_wrap(HOST_MODULE, host::WRITE_FILE, |mut c: Caller<'_, HostState>, p: i32, content: i32| {
            let path = guest_str(&mut c, p)?;
            let text = guest_str(&mut c, content)?;
            let result = c.data().perms.open(&path, true).and_then(|mut file| {
                file.set_len(0).and_then(|_| file.write_all(text.as_bytes())).map_err(|e| format!("{path}: {e}"))
            });
            match result {
                Ok(()) => Ok((1, 0)),
                Err(e) => Ok((0, new_str(&mut c, &e)?)),
            }
        })?;
    }
    if perms.time {
        linker.func_wrap(HOST_MODULE, host::NOW_MS, || {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
        })?;
    }
    if perms.rand {
        linker.func_wrap(HOST_MODULE, host::RAND_INT, |mut c: Caller<'_, HostState>, lo: i64, hi: i64| {
            let span = (hi as u64).wrapping_sub(lo as u64).wrapping_add(1);
            let r = next_random(&mut c.data_mut().rng);
            let offset = if span == 0 { r } else { r % span };
            lo.wrapping_add(offset as i64)
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_permissions() {
        let p = Permissions::parse("io, rand").unwrap();
        assert!(p.allows("io") && p.allows("rand") && !p.allows("fs") && !p.allows("time"));
        let dir = std::env::temp_dir();
        let p = Permissions::parse(&format!("fs:{}", dir.display())).unwrap();
        assert!(p.allows("fs"));
        assert!(Permissions::parse("fs").unwrap_err().contains("needs a directory"));
        assert!(Permissions::parse("net").unwrap_err().contains("unknown effect `net`"));
        assert!(Permissions::parse("fs:/definitely/not/here").is_err());
        assert_eq!(Permissions::parse("").unwrap(), Permissions::default());
    }

    #[test]
    fn fs_paths_must_stay_inside_granted_directories() {
        let root = std::env::temp_dir().join(format!("qal-perm-{}", std::process::id()));
        let inside = root.join("data");
        std::fs::create_dir_all(&inside).unwrap();
        std::fs::write(root.join("secret.txt"), "x").unwrap();
        std::fs::write(inside.join("a.txt"), "a").unwrap();
        let p = Permissions::parse(&format!("fs:{}", inside.display())).unwrap();
        assert!(p.check_path(&inside.join("a.txt").to_string_lossy(), false).is_ok());
        assert!(p.check_path(&inside.join("new.txt").to_string_lossy(), true).is_ok());
        let escape = inside.join("../secret.txt");
        assert!(p.check_path(&escape.to_string_lossy(), false).unwrap_err().starts_with("permission denied"));
        assert!(p.check_path(&root.join("other.txt").to_string_lossy(), true).unwrap_err().starts_with("permission denied"));
        #[cfg(unix)]
        {
            let link = inside.join("link.txt");
            let _ = std::fs::remove_file(&link);
            std::os::unix::fs::symlink(root.join("created-outside.txt"), &link).unwrap();
            assert!(p.check_path(&link.to_string_lossy(), true).unwrap_err().contains("dangling symbolic link"));
            let back = inside.join("back.txt");
            let _ = std::fs::remove_file(&back);
            std::os::unix::fs::symlink(root.join("secret.txt"), &back).unwrap();
            assert!(p.check_path(&back.to_string_lossy(), false).unwrap_err().starts_with("permission denied"));
        }
    }

    #[cfg(unix)]
    #[test]
    fn opening_refuses_final_symlinks_and_detects_swapped_files() {
        let root = std::env::temp_dir().join(format!("qal-open-{}", std::process::id()));
        let inside = root.join("data");
        std::fs::create_dir_all(&inside).unwrap();
        std::fs::write(root.join("secret.txt"), "x").unwrap();
        std::fs::write(inside.join("a.txt"), "a").unwrap();
        let link = inside.join("swap");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(root.join("secret.txt"), &link).unwrap();
        assert!(open_nofollow(&link, false).is_err());
        assert!(open_nofollow(&link, true).is_err());
        assert!(open_nofollow(&inside.join("a.txt"), false).is_ok());
        let p = Permissions::parse(&format!("fs:{}", inside.display())).unwrap();
        let a = open_nofollow(&inside.join("a.txt"), false).unwrap();
        let outside = open_nofollow(&root.join("secret.txt"), false).unwrap();
        let canon = inside.join("a.txt").canonicalize().unwrap();
        assert!(p.verify_opened(&a, &canon).is_ok());
        assert!(p.verify_opened(&outside, &canon).unwrap_err().starts_with("permission denied"));
        assert!(p.verify_opened(&outside, &root.join("secret.txt").canonicalize().unwrap()).unwrap_err().starts_with("permission denied"));
    }

    #[test]
    fn fs_needs_a_known_nofollow_flag() {
        assert_eq!(require_nofollow(Some(0x100)), Ok(0x100));
        let e = require_nofollow(None).unwrap_err();
        assert!(e.starts_with("the `fs` effect is not supported on this platform"), "{e}");
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        assert!(O_NOFOLLOW.is_some(), "supported on the development platforms");
    }

    #[test]
    fn random_numbers_stay_in_range() {
        let mut s = 1;
        for _ in 0..1000 {
            let span = 6u64;
            let v = 1 + (next_random(&mut s) % span) as i64;
            assert!((1..=6).contains(&v));
        }
    }
}

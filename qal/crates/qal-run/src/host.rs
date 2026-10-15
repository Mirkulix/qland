//! Host functions (`qal_host.*`) and the permission model: only granted effect groups are linked,
//! and `fs` only reaches files inside granted directories.

use crate::value::read_str;
use qal_wasm::{host, HOST_MODULE};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use wasmtime::{Caller, Extern, Linker, Memory, StoreLimits};

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

    /// Resolves `path` (relative to the working directory) and checks it lies inside a granted directory.
    /// For writing, the file itself may not exist yet, but its directory must.
    pub fn check_path(&self, path: &str, for_write: bool) -> Result<PathBuf, String> {
        let p = Path::new(path);
        let resolved = if for_write && !p.exists() {
            // A dangling symlink "does not exist" but writing would follow it, possibly outside.
            if std::fs::symlink_metadata(p).is_ok() {
                return Err(format!("permission denied: `{path}` is a dangling symbolic link"));
            }
            let parent = match p.parent() {
                Some(d) if !d.as_os_str().is_empty() => d,
                _ => Path::new("."),
            };
            let name = p.file_name().ok_or_else(|| format!("`{path}` is not a file path"))?;
            parent.canonicalize().map_err(|e| format!("{path}: {e}"))?.join(name)
        } else {
            p.canonicalize().map_err(|e| format!("{path}: {e}"))?
        };
        if self.fs.iter().any(|root| resolved.starts_with(root)) {
            Ok(resolved)
        } else {
            Err(format!("permission denied: `{path}` is outside the allowed directories"))
        }
    }
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

/// Allocates a Str in guest memory via the exported allocator (which may fail with out_of_memory).
pub fn new_str(c: &mut Caller<'_, HostState>, s: &str) -> Result<i32, wasmtime::Error> {
    let alloc = c.get_export("qal_alloc").and_then(Extern::into_func).ok_or_else(|| wasmtime::Error::msg("no qal_alloc"))?;
    let p = alloc.typed::<i64, i32>(&c)?.call(&mut *c, 4 + s.len() as i64)?;
    let mem = memory(c)?;
    mem.write(&mut *c, p as usize, &(s.len() as u32).to_le_bytes())?;
    mem.write(&mut *c, p as usize + 4, s.as_bytes())?;
    Ok(p)
}

/// splitmix64
fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Defines `fail`, the pure helpers, and the functions of every granted effect.
pub fn link(linker: &mut Linker<HostState>, perms: &Permissions) -> Result<(), wasmtime::Error> {
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
            let out = &mut c.data_mut().stdout;
            out.extend_from_slice(s.as_bytes());
            out.push(b'\n');
            Ok(())
        })?;
        linker.func_wrap(HOST_MODULE, host::READ_LINE, |mut c: Caller<'_, HostState>| {
            let mut buf = Vec::new();
            let n = c.data_mut().stdin.read_until(b'\n', &mut buf).map_err(wasmtime::Error::msg)?;
            if n == 0 {
                return Ok(-1);
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
            let result = c.data().perms.check_path(&path, false).and_then(|full| {
                let bytes = std::fs::read(&full).map_err(|e| format!("{path}: {e}"))?;
                String::from_utf8(bytes).map_err(|_| format!("{path}: not valid UTF-8"))
            });
            let (ok, text) = match result {
                Ok(s) => (1, s),
                Err(e) => (0, e),
            };
            Ok((ok, new_str(&mut c, &text)?))
        })?;
        linker.func_wrap(HOST_MODULE, host::WRITE_FILE, |mut c: Caller<'_, HostState>, p: i32, content: i32| {
            let path = guest_str(&mut c, p)?;
            let text = guest_str(&mut c, content)?;
            let result = c.data().perms.check_path(&path, true).and_then(|full| std::fs::write(&full, text).map_err(|e| format!("{path}: {e}")));
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

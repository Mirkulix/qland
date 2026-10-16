//! `qal team ...` and `qal verify` (Phase 4 spec §4.1, §9.2): the owner's command line for a team
//! store. Every result is JSON; secrets (keys, session tokens) are written only to their files
//! (0600) and never printed.

use crate::ops::{Ctx, Output, Reply};
use qal_core::diag::Diagnostic;
use qal_store::team::keys::KeyDir;
use qal_store::team::policy::Role;
use qal_store::team::verify::Trust;
use qal_store::Store;
use serde::Serialize;
use serde_json::{json, Value};
use std::path::Path;

/// Printed by `qal team init` (spec §2 decision 15).
pub const THREAT_MODEL: &str = "roles are guard rails for cooperative agents, not a security boundary: every agent runs as the same OS user and could read the key files; the signed records make every change of the store evident (`qal verify --trust team.json`)";

fn reply(fields: Value) -> Reply {
    let mut out = Output::default();
    if let Value::Object(m) = fields {
        out.extra = m;
    }
    Reply::new(out, 0)
}

fn value<T: Serialize>(x: &T) -> Value {
    serde_json::to_value(x).unwrap_or(Value::Null)
}

fn keys() -> Result<KeyDir, Vec<Diagnostic>> {
    KeyDir::from_env().map_err(|d| vec![d])
}

fn store(ctx: &Ctx) -> Result<Store, Vec<Diagnostic>> {
    ctx.open()
}

fn run(f: impl FnOnce() -> Result<Reply, Vec<Diagnostic>>) -> Reply {
    f().unwrap_or_else(Reply::diagnostics)
}

/// `qal team init [--owner <id>]`: imports the standard library first (as any write before team
/// mode), then makes the store a team store with the owner's key and the genesis record.
pub fn init(ctx: &Ctx, owner: &str) -> Reply {
    run(|| {
        let keys = keys()?;
        let mut s = store(ctx)?;
        let std = crate::ops::with_std(&mut s);
        let i = s.team_init(owner, &keys)?;
        let mut r = reply(json!({"team": i, "note": THREAT_MODEL}));
        std.report(&mut r.out);
        Ok(r)
    })
}

pub fn add(ctx: &Ctx, id: &str, role: &str) -> Reply {
    run(|| {
        let Some(role) = Role::parse(role) else {
            return Err(vec![Diagnostic::new("E0701", "bad_arguments", "", "", format!("`{role}` is not a role")).hint("roles: architect, implementer, verifier, reviewer")]);
        };
        let keys = keys()?;
        let mut s = store(ctx)?;
        let owner = s.owner_signer(&keys)?;
        Ok(reply(json!({"member": s.team_add(&owner, id, role, &keys)?})))
    })
}

pub fn rekey(ctx: &Ctx, id: &str) -> Reply {
    run(|| {
        let keys = keys()?;
        let mut s = store(ctx)?;
        let owner = s.owner_signer(&keys)?;
        Ok(reply(json!({"member": s.team_rekey(&owner, id, &keys)?})))
    })
}

pub fn revoke(ctx: &Ctx, id: &str) -> Reply {
    run(|| {
        let keys = keys()?;
        let mut s = store(ctx)?;
        let owner = s.owner_signer(&keys)?;
        Ok(reply(json!({"revoked": s.team_revoke(&owner, id)?})))
    })
}

/// `qal team session <id> [--ttl 24h] --out <file>`: the token goes only into `out` (0600).
pub fn session(ctx: &Ctx, id: &str, ttl: Option<&str>, out: &Path) -> Reply {
    run(|| {
        let ttl_ms = match ttl {
            None => qal_store::team::DEFAULT_SESSION_TTL_MS,
            Some(t) => qal_store::team::parse_ttl(t).ok_or_else(|| vec![Diagnostic::new("E0701", "bad_arguments", "", "", format!("`{t}` is not a duration")).hint("e.g. 90m, 24h, 7d")])?,
        };
        let mut s = store(ctx)?;
        // Refuse a link, a directory or a place in the project before a session is made.
        crate::secret_file::check_target(out)?;
        if qal_store::team::keys::in_project(out, s.project_dir()?).map_err(|d| vec![d])? {
            return Err(vec![Diagnostic::new("E0926", "key_dir_in_project", "", "", format!("the session file {} would be inside the project", out.display()))
                .hint("a session token is a secret: write it outside the project (next to the keys, e.g. ~/.config/qal/sessions)")]);
        }
        let new = s.team_session(id, ttl_ms)?;
        // Reserved up front: the buffer with the token never reallocates.
        let mut line = zeroize::Zeroizing::new(String::with_capacity(new.token.len() + 1));
        line.push_str(&new.token);
        line.push('\n');
        crate::secret_file::write(out, line.as_bytes())?;
        Ok(reply(json!({"session": {"id": new.id, "member": new.member, "expires_at": new.expires_at, "file": out.display().to_string()}})))
    })
}

/// `qal team export-trust --out team.json`: public keys only.
pub fn export_trust(ctx: &Ctx, out: &Path) -> Reply {
    run(|| {
        let s = store(ctx)?;
        let trust = s.team_trust()?;
        let text = serde_json::to_string_pretty(&trust).map_err(|e| vec![Diagnostic::store_error(format!("cannot write the trust file: {e}"))])? + "\n";
        std::fs::write(out, text).map_err(|e| vec![Diagnostic::store_error(format!("cannot write {}: {e}", out.display()))])?;
        Ok(reply(json!({"trust": {"file": out.display().to_string(), "owner": trust["owner"], "members": trust["members"].as_array().map_or(0, Vec::len)}})))
    })
}

pub fn status(ctx: &Ctx) -> Reply {
    run(|| Ok(reply(json!({"status": store(ctx)?.team_status()?}))))
}

pub fn stats(ctx: &Ctx) -> Reply {
    run(|| Ok(reply(json!({"stats": store(ctx)?.team_stats()?}))))
}

/// `qal team sync-std`: imports the standard library embedded in this qal as a signed `import`
/// record of the owner (team stores never import it automatically).
pub fn sync_std(ctx: &Ctx) -> Reply {
    run(|| {
        let keys = keys()?;
        let mut s = store(ctx)?;
        let owner = s.owner_signer(&keys)?;
        s.set_release(crate::STD_PACKAGE)?;
        let embedded = qal_store::package::parse(crate::STD_PACKAGE)?.hash;
        if s.meta(qal_store::package::META_STD_PACKAGE)?.as_deref() == Some(embedded.as_str()) {
            return Ok(reply(json!({"std": {"hash": embedded, "status": "up_to_date"}})));
        }
        s.set_signer(Some(std::sync::Arc::new(owner)));
        let (result, _) = crate::ops::proving(|v| s.import_verified(crate::STD_PACKAGE, v));
        let imported = result?;
        Ok(reply(json!({"std": {"hash": imported.hash, "status": "imported", "defined": imported.defined.len()}})))
    })
}

/// `qal verify [--trust team.json] [--from <seq>]` (and `qal_verify`): exit 1 unless ok.
pub fn verify(ctx: &Ctx, trust: Option<&str>, from: i64, head: Option<&str>) -> Reply {
    run(|| {
        let trust = trust.map(Trust::parse).transpose().map_err(|d| vec![d])?;
        let outcome = store(ctx)?.verify(trust.as_ref(), from, head)?;
        let ok = outcome.ok;
        let mut r = reply(value(&outcome));
        if !ok {
            r.out.failure = true;
            r.code = 1;
        }
        Ok(r)
    })
}

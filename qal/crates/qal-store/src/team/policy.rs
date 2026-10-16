//! The rights matrix of Phase 4 spec §8: one table, read by the MCP server (calls and
//! `tools/list`), its error messages, the store (which refuses unsigned or unauthorized writes) and
//! `qal verify` (which re-checks every record).

use qal_core::diag::Diagnostic;

/// An agent role (the owner acts only on the command line).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    Architect,
    Implementer,
    Verifier,
    Reviewer,
}

pub const ROLES: [Role; 4] = [Role::Architect, Role::Implementer, Role::Verifier, Role::Reviewer];

/// The role name of the team's owner (in `members.role` and records).
pub const OWNER: &str = "owner";

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Architect => "architect",
            Role::Implementer => "implementer",
            Role::Verifier => "verifier",
            Role::Reviewer => "reviewer",
        }
    }

    pub fn parse(s: &str) -> Option<Role> {
        ROLES.into_iter().find(|r| r.as_str() == s)
    }

    fn index(self) -> usize {
        match self {
            Role::Architect => 0,
            Role::Implementer => 1,
            Role::Verifier => 2,
            Role::Reviewer => 3,
        }
    }
}

/// Where an operation works (§6.1): the main state or a change set, seen from the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Space {
    Main,
    /// A change set the caller owns.
    Own,
    /// Another member's change set with status `submitted`.
    ForeignSubmitted,
    /// Another member's change set in any other status.
    ForeignOther,
}

pub const SPACES: [Space; 4] = [Space::Main, Space::Own, Space::ForeignSubmitted, Space::ForeignOther];

/// One cell of the matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// ✓ anywhere.
    Yes,
    /// ✗ (E0903).
    No,
    /// „eigen“: only in the caller's own change set.
    Own,
    /// „eigen + Beitrag“: the own change set or a submitted foreign one (the verifier's tests).
    OwnOrContribution,
    /// Any change set (the architect's `abandon`).
    AnyChange,
}

impl Rule {
    fn allows(self, space: Space) -> bool {
        match self {
            Rule::Yes => true,
            Rule::No => false,
            // `main` changes only by merge (Phase 4c): code is written in change sets.
            Rule::Own => space == Space::Own,
            Rule::OwnOrContribution => matches!(space, Space::Own | Space::ForeignSubmitted),
            Rule::AnyChange => space != Space::Main,
        }
    }

    /// Allowed in at least one space.
    fn ever(self) -> bool {
        self != Rule::No
    }
}

/// One row: a tool, optionally one of its actions, and the rule per role (architect,
/// implementer, verifier, reviewer). `available`: the tool exists in this build (the others come
/// with Phase 4c and are part of the table already).
#[derive(Debug, Clone, Copy)]
pub struct Row {
    pub tool: &'static str,
    pub action: Option<&'static str>,
    pub rules: [Rule; 4],
    pub available: bool,
}

use Rule::{AnyChange, No, Own, OwnOrContribution, Yes};

const fn row(tool: &'static str, action: Option<&'static str>, rules: [Rule; 4], available: bool) -> Row {
    Row { tool, action, rules, available }
}

/// The matrix §8 (plus `qal_delete`, which Phase 3b added after the spec: like `qal_rename`).
pub const POLICY: &[Row] = &[
    row("qal_view", None, [Yes, Yes, Yes, Yes], true),
    row("qal_query", None, [Yes, Yes, Yes, Yes], true),
    row("qal_history", None, [Yes, Yes, Yes, Yes], true),
    row("qal_whoami", None, [Yes, Yes, Yes, Yes], true),
    row("qal_verify", None, [Yes, Yes, Yes, Yes], true),
    row("qal_test", None, [Yes, Yes, Yes, Yes], true),
    row("qal_prove", None, [Yes, Yes, Yes, Yes], true),
    row("qal_run", None, [Yes, Yes, Yes, Yes], true),
    row("qal_call", None, [Yes, Yes, Yes, Yes], true),
    row("qal_request", None, [Yes, Yes, Yes, Yes], true),
    row("qal_build", None, [Yes, Yes, Yes, Yes], true),
    row("qal_define", None, [Own, Own, OwnOrContribution, No], true),
    row("qal_rename", None, [Own, No, No, No], true),
    row("qal_delete", None, [Own, No, No, No], true),
    row("qal_lease", Some("acquire"), [Yes, Yes, Yes, No], true),
    row("qal_lease", Some("renew"), [Yes, Yes, Yes, No], true),
    row("qal_lease", Some("release"), [Yes, Yes, Yes, No], true),
    row("qal_lease", Some("break"), [Yes, No, No, No], true),
    row("qal_change", Some("list"), [Yes, Yes, Yes, Yes], true),
    row("qal_change", Some("show"), [Yes, Yes, Yes, Yes], true),
    row("qal_change", Some("diff"), [Yes, Yes, Yes, Yes], true),
    row("qal_change", Some("open"), [Yes, Yes, Yes, No], true),
    row("qal_change", Some("claim"), [No, Yes, Yes, No], true),
    row("qal_change", Some("submit"), [Own, Own, Own, No], true),
    row("qal_change", Some("abandon"), [AnyChange, Own, Own, No], true),
    row("qal_change", Some("scope"), [Yes, No, No, No], true),
    row("qal_rebase", None, [Own, Own, Own, No], false),
    row("qal_review", Some("comment"), [Yes, No, No, No], false),
    row("qal_review", Some("verified"), [No, No, Yes, No], false),
    row("qal_review", Some("failed"), [No, No, Yes, No], false),
    row("qal_review", Some("approve"), [No, No, No, Yes], false),
    row("qal_review", Some("changes"), [No, No, No, Yes], false),
    row("qal_merge", None, [Yes, No, No, Yes], false),
];

/// The tools that write code (and therefore need a signing key).
pub const WRITE_TOOLS: [&str; 3] = ["qal_define", "qal_rename", "qal_delete"];

/// The rows of `tool` (with any action).
fn rows(tool: &str) -> impl Iterator<Item = &'static Row> + '_ {
    POLICY.iter().filter(move |r| r.tool == tool)
}

/// The rule for `tool`/`action` and `role`, if the matrix has the row.
pub fn rule(tool: &str, action: Option<&str>, role: Role) -> Option<Rule> {
    rows(tool).find(|r| r.action == action).map(|r| r.rules[role.index()])
}

/// May `role` call `tool` (with `action`) in `space`?
pub fn allows(tool: &str, action: Option<&str>, role: Role, space: Space) -> bool {
    rule(tool, action, role).is_some_and(|r| r.allows(space))
}

/// The roles that may call `tool`/`action` in `space`.
pub fn allowed_roles(tool: &str, action: Option<&str>, space: Space) -> Vec<Role> {
    ROLES.into_iter().filter(|r| allows(tool, action, *r, space)).collect()
}

/// The available tools `role` may call in some space with some action: what `tools/list` shows.
pub fn tools_for(role: Role) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for r in POLICY.iter().filter(|r| r.available && r.rules[role.index()].ever()) {
        if !out.contains(&r.tool) {
            out.push(r.tool);
        }
    }
    out
}

/// E0903 `role_forbidden` for `role` calling `tool`/`action` in `space`, naming the roles that may.
pub fn forbidden(tool: &str, action: Option<&str>, role: Role, space: Space) -> Diagnostic {
    let what = match action {
        Some(a) => format!("`{tool}` with action `{a}`"),
        None => format!("`{tool}`"),
    };
    let expected = allowed_roles(tool, action, space);
    let names: Vec<&str> = expected.iter().map(|r| r.as_str()).collect();
    let anywhere: Vec<&str> = ROLES.into_iter().filter(|r| rule(tool, action, *r).is_some_and(Rule::ever)).map(Role::as_str).collect();
    let place = match space {
        Space::Main => "on main",
        Space::Own => "in an own change set",
        Space::ForeignSubmitted | Space::ForeignOther => "in another member's change set",
    };
    let mut d = Diagnostic::new("E0903", "role_forbidden", "", "", format!("{what} is not allowed for the role {} {place}", role.as_str()));
    d.expected = Some(if names.is_empty() { "nobody".into() } else { names.join(", ") });
    d.found = Some(role.as_str().into());
    let hint = if space == Space::Main && rule(tool, action, role).is_some_and(|r| r == Rule::Own || r == Rule::OwnOrContribution) {
        format!("{}s write only in their own change set: open one (qal_change {{\"action\": \"open\"}}) or claim one, then pass `change` (or it becomes your current one); main changes only by merge", role.as_str())
    } else if names.is_empty() && anywhere.is_empty() {
        "no agent role may do this; the owner can, on the command line".to_string()
    } else if names.is_empty() {
        format!("allowed for {} elsewhere; ask a member with that role", anywhere.join(", "))
    } else {
        format!("ask a member with the role {} to do it (see qal_whoami for your role)", names.join(" or "))
    };
    d.hint(hint)
}

/// What a record of operation `op` (with `action`, for `change`) in a space of kind `space` needs
/// from its signer (checked when a record is written and again by `qal verify`).
pub fn record_allowed(op: &str, action: Option<&str>, role: &str, space: Space) -> bool {
    let agent = |tool: &str, action: Option<&str>| Role::parse(role).is_some_and(|r| allows(tool, action, r, space));
    match op {
        "genesis" | "member_add" | "member_key" | "member_revoke" | "import" => role == OWNER,
        "define" | "rename" | "delete" => agent(&format!("qal_{op}"), None),
        "change" => action.is_some_and(|a| a != "list" && a != "show" && a != "diff" && agent("qal_change", Some(a))),
        "lease_break" => agent("qal_lease", Some("break")),
        _ => false,
    }
}

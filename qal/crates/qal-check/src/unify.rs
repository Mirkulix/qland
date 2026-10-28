//! Type variables and unification for local type inference inside one definition.
//!
//! Inferred types can nest far deeper than the AST (each `let` may wrap the previous value), and
//! because variables are shared they can also grow exponentially in size (`if c then Ok(a) else
//! Err(a)` doubles the type of `a`). The depth and the size (number of nodes when fully expanded) of
//! every variable's binding are tracked exactly and capped at [`MAX_TYPE_DEPTH`] and
//! [`MAX_TYPE_SIZE`]; all later work on types (resolution, display, equality, decoding) is bounded
//! by them. Variables are only ever resolved shallowly, so no operation re-walks a whole nested
//! type at every level.

use qal_core::ast::Type;
use std::collections::{BTreeSet, HashSet};

/// Maximum nesting depth of an inferred type.
pub const MAX_TYPE_DEPTH: u32 = 256;
/// Maximum size (nodes of the fully expanded tree) of an inferred type.
pub const MAX_TYPE_SIZE: u32 = 4096;

/// An effect variable: the effects it must at least carry (names, or `?M` for other variables),
/// and whether it may still grow. It grows while it stands on the expected side of a check
/// (`found ⊆ expected`): each argument adds what it needs, so one parameter can serve several
/// callbacks with the union of their effects. It is frozen as soon as a check depends on its
/// value staying as it is (it stood on the found side, or a call's effects were checked), so a
/// later growth can never invalidate an earlier check.
#[derive(Debug, Default)]
struct Evar {
    lower: Vec<String>,
    frozen: bool,
}

#[derive(Debug, Default)]
pub struct Subst {
    /// Binding of each inference variable `Type::Var(i)`.
    vars: Vec<Option<Type>>,
    /// Depth of each variable's type (1 while unbound), kept exact as inner variables get bound.
    depth: Vec<u32>,
    /// Size of each variable's type (1 while unbound), saturating just above `MAX_TYPE_SIZE`.
    size: Vec<u32>,
    /// For each variable, the variables whose binding mentions it directly.
    users: Vec<Vec<u32>>,
    /// Effect variables: an entry `?N` in the effect list of a function type stands for variable
    /// `N`. A call of a function with an effect parameter makes one; the arguments grow it
    /// (`unify_effects`) until something reads its value, which freezes it.
    evars: Vec<Evar>,
    /// Set when a unification failed because a type would exceed `MAX_TYPE_DEPTH`.
    pub too_deep: bool,
    /// Set when a unification failed because a type would exceed `MAX_TYPE_SIZE`.
    pub too_large: bool,
}

impl Subst {
    pub fn fresh(&mut self) -> Type {
        self.vars.push(None);
        self.depth.push(1);
        self.size.push(1);
        self.users.push(Vec::new());
        Type::Var(self.vars.len() as u32 - 1)
    }

    /// A variable bound to `t` (with unbound variables inside), so that `t`'s depth stays tracked
    /// as those get bound. Used for constructed List/Option/Result types.
    pub fn wrap(&mut self, t: Type) -> Type {
        let v = self.fresh();
        let ok = self.unify(&v, &t);
        debug_assert!(ok, "binding a fresh variable");
        v
    }

    /// Follows variable bindings at the top level only. Unbound variables stay.
    pub fn resolve(&self, t: &Type) -> Type {
        let mut t = t;
        while let Type::Var(v) = t {
            match &self.vars[*v as usize] {
                Some(b) => t = b,
                None => break,
            }
        }
        t.clone()
    }

    /// Replaces all bound variables; unbound ones become `Unit` (their representation never matters).
    /// Effect lists become their concrete effects (an open effect variable adds none).
    pub fn finish(&self, t: &Type) -> Type {
        match self.resolve(t) {
            Type::Var(_) => Type::Unit,
            Type::List(x) => Type::list(self.finish(&x)),
            Type::Option(x) => Type::option(self.finish(&x)),
            Type::Result(a, b) => Type::result(self.finish(&a), self.finish(&b)),
            Type::Map(a, b) => Type::map(self.finish(&a), self.finish(&b)),
            Type::App(n, args) => Type::App(n, args.iter().map(|a| self.finish(a)).collect()),
            Type::Fn(ps, r, e) => Type::Fn(ps.iter().map(|p| self.finish(p)).collect(), Box::new(self.finish(&r)), self.resolve_effects(&e).0.into_iter().collect()),
            t => t,
        }
    }

    /// A fresh effect variable, as the entry a function type's effect list carries for it.
    pub fn fresh_effect(&mut self) -> String {
        self.evars.push(Evar::default());
        format!("?{}", self.evars.len() - 1)
    }

    fn evar(e: &str) -> Option<u32> {
        e.strip_prefix('?').and_then(|n| n.parse().ok())
    }

    /// The effects an effect list carries (following variables) and the variables in it that may
    /// still grow.
    pub fn resolve_effects(&self, effs: &[String]) -> (BTreeSet<String>, BTreeSet<u32>) {
        let (mut names, mut open, mut seen) = (BTreeSet::new(), BTreeSet::new(), HashSet::new());
        let mut stack: Vec<String> = effs.to_vec();
        while let Some(e) = stack.pop() {
            match Self::evar(&e) {
                Some(v) if !seen.insert(v) => {}
                Some(v) => {
                    let ev = &self.evars[v as usize];
                    stack.extend(ev.lower.iter().cloned());
                    if !ev.frozen {
                        open.insert(v);
                    }
                }
                None => {
                    names.insert(e);
                }
            }
        }
        (names, open)
    }

    /// Freezes every growable variable in `effs`: its value has been read.
    pub fn freeze_effects(&mut self, effs: &[String]) {
        for v in self.resolve_effects(effs).1 {
            self.evars[v as usize].frozen = true;
        }
    }

    /// `found ⊆ expected` as effect sets (`exact`: equal). A growable variable on the expected
    /// side takes what `found` has beyond `expected` (the union over all arguments of a call);
    /// without one, the check is on the effects as they are, and the variables of `found` are
    /// frozen (the check depends on them).
    pub fn unify_effects(&mut self, expected: &[String], found: &[String], exact: bool) -> bool {
        self.effects_sub(expected, found) && (!exact || self.effects_sub(found, expected))
    }

    fn effects_sub(&mut self, expected: &[String], found: &[String]) -> bool {
        let (c1, v1) = self.resolve_effects(expected);
        let (c2, v2) = self.resolve_effects(found);
        if let Some(&v) = v1.iter().next() {
            let extra: Vec<String> = c2.difference(&c1).cloned().chain(v2.iter().filter(|w| !v1.contains(w)).map(|w| format!("?{w}"))).collect();
            let lower = &mut self.evars[v as usize].lower;
            for e in extra {
                if !lower.contains(&e) {
                    lower.push(e);
                }
            }
            return true;
        }
        if !c2.is_subset(&c1) {
            return false;
        }
        for w in v2 {
            self.evars[w as usize].frozen = true;
        }
        true
    }

    /// Depth of `t`, using the tracked depth of its variables (recursion only over `t` as written).
    fn depth_of(&self, t: &Type) -> u32 {
        match t {
            Type::Var(v) => self.depth[*v as usize],
            Type::List(x) | Type::Option(x) => 1 + self.depth_of(x),
            Type::Result(a, b) | Type::Map(a, b) => 1 + self.depth_of(a).max(self.depth_of(b)),
            Type::App(_, args) => 1 + args.iter().map(|a| self.depth_of(a)).max().unwrap_or(0),
            Type::Fn(ps, r, _) => 1 + ps.iter().map(|p| self.depth_of(p)).max().unwrap_or(0).max(self.depth_of(r)),
            _ => 1,
        }
    }

    /// Size of `t`, using the tracked size of its variables; saturates at `MAX_TYPE_SIZE + 1`.
    fn size_of(&self, t: &Type) -> u32 {
        let n = match t {
            Type::Var(v) => self.size[*v as usize],
            Type::List(x) | Type::Option(x) => 1 + self.size_of(x),
            Type::Result(a, b) | Type::Map(a, b) => (1 + self.size_of(a)).saturating_add(self.size_of(b)),
            Type::App(_, args) => args.iter().fold(1u32, |n, a| n.saturating_add(self.size_of(a))),
            Type::Fn(ps, r, _) => ps.iter().fold(1 + self.size_of(r), |n, p| n.saturating_add(self.size_of(p))),
            _ => 1,
        };
        n.min(MAX_TYPE_SIZE + 1)
    }

    /// Records a limit violation; true if `(depth, size)` is within the limits.
    fn within_limits(&mut self, depth: u32, size: u32) -> bool {
        if depth > MAX_TYPE_DEPTH {
            self.too_deep = true;
            return false;
        }
        if size > MAX_TYPE_SIZE {
            self.too_large = true;
            return false;
        }
        true
    }

    /// The variables `t` mentions directly (as written).
    fn vars_in(t: &Type, out: &mut Vec<u32>) {
        match t {
            Type::Var(v) => out.push(*v),
            Type::List(x) | Type::Option(x) => Self::vars_in(x, out),
            Type::Result(a, b) | Type::Map(a, b) => {
                Self::vars_in(a, out);
                Self::vars_in(b, out);
            }
            Type::App(_, args) => args.iter().for_each(|a| Self::vars_in(a, out)),
            Type::Fn(ps, r, _) => ps.iter().chain([&**r]).for_each(|p| Self::vars_in(p, out)),
            _ => {}
        }
    }

    /// Binds unbound `x` to `t` and updates the depth and size of every binding that (transitively)
    /// mentions `x`. Fails, setting `too_deep` or `too_large`, if any of them would exceed a limit.
    fn bind(&mut self, x: u32, t: Type) -> bool {
        let (d, n) = (self.depth_of(&t), self.size_of(&t));
        if !self.within_limits(d, n) {
            return false;
        }
        let mut direct = Vec::new();
        Self::vars_in(&t, &mut direct);
        for w in direct {
            self.users[w as usize].push(x);
        }
        self.vars[x as usize] = Some(t);
        self.depth[x as usize] = d;
        self.size[x as usize] = n;
        // Depths and sizes only grow and are capped, so each variable is revisited a bounded number of times.
        let mut changed = vec![x];
        while let Some(v) = changed.pop() {
            for k in 0..self.users[v as usize].len() {
                let u = self.users[v as usize][k];
                // Users are always bound (they are recorded when bound); skip one that is not.
                let Some(bound) = self.vars[u as usize].as_ref() else { continue };
                let (nd, nn) = (self.depth_of(bound), self.size_of(bound));
                if nd > self.depth[u as usize] || nn > self.size[u as usize] {
                    if !self.within_limits(nd, nn) {
                        return false;
                    }
                    self.depth[u as usize] = nd;
                    self.size[u as usize] = nn;
                    changed.push(u);
                }
            }
        }
        true
    }

    /// True if variable `v` occurs in `t` (`seen` skips variables already searched).
    fn occurs(&self, v: u32, t: &Type, seen: &mut HashSet<u32>) -> bool {
        match t {
            Type::Var(w) if *w == v => true,
            Type::Var(w) => {
                if !seen.insert(*w) {
                    return false;
                }
                self.vars[*w as usize].as_ref().is_some_and(|b| self.occurs(v, b, seen))
            }
            Type::List(x) | Type::Option(x) => self.occurs(v, x, seen),
            Type::Result(a, b) | Type::Map(a, b) => self.occurs(v, a, seen) || self.occurs(v, b, seen),
            Type::App(_, args) => args.iter().any(|a| self.occurs(v, a, seen)),
            Type::Fn(ps, r, _) => ps.iter().any(|p| self.occurs(v, p, seen)) || self.occurs(v, r, seen),
            _ => false,
        }
    }

    /// Makes `expected` and `found` equal. A function value may have fewer effects than expected.
    /// On failure, bindings made so far are kept (the caller reports a diagnostic and stops).
    ///
    /// Effect subtyping goes through List/Option/Result/Map (immutable, no function-parameter
    /// position) covariantly and through function types with flipped parameters. The arguments of a
    /// user generic type are invariant: `type Sink[T] = { put: fn(T) -> Unit }` uses `T` in a
    /// parameter position, so `Sink[fn() -> Unit]` must not pass as `Sink[fn() -> Unit !io]`.
    pub fn unify(&mut self, expected: &Type, found: &Type) -> bool {
        self.unify_in(expected, found, false)
    }

    /// Makes `a` and `b` equal, effects of function types included (as sets).
    pub fn unify_exact(&mut self, a: &Type, b: &Type) -> bool {
        self.unify_in(a, b, true)
    }

    /// Unification visits every node of both types once: an invariant position is unified once in
    /// `exact` mode (equal effect sets), not once in each direction, which would cost 2^depth for
    /// nested type applications such as `B[B[B[Int]]]`.
    fn unify_in(&mut self, expected: &Type, found: &Type, exact: bool) -> bool {
        // A variable unifies with itself (without resolving: that clones its binding's top node).
        if let (Type::Var(x), Type::Var(y)) = (expected, found) {
            if x == y {
                return true;
            }
        }
        let (a, b) = (self.resolve(expected), self.resolve(found));
        if a == b && !Self::has_vars(&a) {
            return true;
        }
        match (&a, &b) {
            (Type::Var(x), Type::Var(y)) if x == y => true,
            (Type::Var(x), t) | (t, Type::Var(x)) => {
                if self.occurs(*x, t, &mut HashSet::new()) {
                    return false;
                }
                self.bind(*x, t.clone())
            }
            (Type::List(x), Type::List(y)) | (Type::Option(x), Type::Option(y)) => self.unify_in(x, y, exact),
            (Type::Result(a1, b1), Type::Result(a2, b2)) | (Type::Map(a1, b1), Type::Map(a2, b2)) => self.unify_in(a1, a2, exact) && self.unify_in(b1, b2, exact),
            (Type::App(n1, x1), Type::App(n2, x2)) => n1 == n2 && x1.len() == x2.len() && x1.iter().zip(x2).all(|(a, b)| self.unify_in(a, b, true)),
            (Type::Fn(p1, r1, e1), Type::Fn(p2, r2, e2)) => {
                p1.len() == p2.len()
                    // Parameters are contravariant: effect subtyping flips.
                    && p1.iter().zip(p2).all(|(x, y)| self.unify_in(y, x, exact))
                    && self.unify_in(r1, r2, exact)
                    && self.unify_effects(e1, e2, exact)
            }
            _ => a == b,
        }
    }

    /// True if `t` (as written) mentions an inference variable.
    fn has_vars(t: &Type) -> bool {
        match t {
            Type::Var(_) => true,
            Type::List(x) | Type::Option(x) => Self::has_vars(x),
            Type::Result(a, b) | Type::Map(a, b) => Self::has_vars(a) || Self::has_vars(b),
            Type::App(_, args) => args.iter().any(Self::has_vars),
            Type::Fn(ps, r, _) => ps.iter().any(Self::has_vars) || Self::has_vars(r),
            _ => false,
        }
    }

    /// True if a value of type `t` (through bound variables) has a function type at a position
    /// where effects join: inside List/Option/Result/Map and function types. Generic type
    /// arguments are invariant, so they are not searched. Linear in the distinct nodes of `t`.
    pub fn has_join_fn(&self, t: &Type) -> bool {
        self.join_fn_in(t, &mut HashSet::new())
    }

    fn join_fn_in(&self, t: &Type, seen: &mut HashSet<u32>) -> bool {
        match t {
            Type::Var(v) => seen.insert(*v) && self.vars[*v as usize].as_ref().is_some_and(|b| self.join_fn_in(b, seen)),
            Type::Fn(..) => true,
            Type::List(x) | Type::Option(x) => self.join_fn_in(x, seen),
            Type::Result(a, b) | Type::Map(a, b) => self.join_fn_in(a, seen) || self.join_fn_in(b, seen),
            _ => false,
        }
    }

    /// The least type both `a` and `b` can be used as (`covariant`), or the greatest type usable as
    /// both (`!covariant`, for function parameters): function types join their effects (union, or
    /// intersection for parameters) through List/Option/Result/Map and function results and
    /// parameters; every other position is unified exactly. So `if c then fn() => () else p` with
    /// `p: fn() -> Unit !io` has the type of `p` whichever branch comes first. None if the types
    /// do not fit together (bindings made so far are kept, as for `unify`).
    ///
    /// Where the join equals `a`, `a` itself is returned (as written, usually a variable), so
    /// repeated joins (one per list item) keep types shared instead of rebuilding them as trees.
    pub fn join(&mut self, a: &Type, b: &Type, covariant: bool) -> Option<Type> {
        self.join_in(a, b, covariant).map(|(t, _)| t)
    }

    /// The join and whether it differs from `a`.
    fn join_in(&mut self, a: &Type, b: &Type, covariant: bool) -> Option<(Type, bool)> {
        let (ra, rb) = (self.resolve(a), self.resolve(b));
        if let (Type::Var(x), Type::Var(y)) = (a, b) {
            if x == y {
                return Some((a.clone(), false));
            }
        }
        // A changed node is bound to a fresh variable, so that the joined type stays a chain of
        // variables: `resolve` (one clone of the top node per level) stays cheap on it.
        let same = |s: &mut Self, t: Type, changed: bool| Some(if changed { (s.wrap(t), true) } else { (a.clone(), false) });
        match (&ra, &rb) {
            (Type::List(x), Type::List(y)) => {
                let (j, c) = self.join_in(x, y, covariant)?;
                same(self, Type::list(j), c)
            }
            (Type::Option(x), Type::Option(y)) => {
                let (j, c) = self.join_in(x, y, covariant)?;
                same(self, Type::option(j), c)
            }
            (Type::Result(a1, b1), Type::Result(a2, b2)) => {
                let ((ja, ca), (jb, cb)) = (self.join_in(a1, a2, covariant)?, self.join_in(b1, b2, covariant)?);
                same(self, Type::result(ja, jb), ca || cb)
            }
            (Type::Map(a1, b1), Type::Map(a2, b2)) => {
                let ((ja, ca), (jb, cb)) = (self.join_in(a1, a2, covariant)?, self.join_in(b1, b2, covariant)?);
                same(self, Type::map(ja, jb), ca || cb)
            }
            (Type::Fn(p1, r1, e1), Type::Fn(p2, r2, e2)) if p1.len() == p2.len() => {
                let mut ps = Vec::with_capacity(p1.len());
                let mut changed = false;
                for (x, y) in p1.iter().zip(p2) {
                    let (j, c) = self.join_in(x, y, !covariant)?;
                    changed |= c;
                    ps.push(j);
                }
                let (r, c) = self.join_in(r1, r2, covariant)?;
                // Open effect variables stay in a union (the join may still grow) and leave an
                // intersection (a parameter usable as both is pure where nothing is known).
                let ((c1, v1), (c2, v2)) = (self.resolve_effects(e1), self.resolve_effects(e2));
                let effs: Vec<String> = if covariant {
                    c1.union(&c2).cloned().chain(v1.union(&v2).map(|v| format!("?{v}"))).collect()
                } else {
                    c1.intersection(&c2).cloned().collect()
                };
                let changed = changed || c || effs.len() != e1.len() || !effs.iter().all(|e| e1.contains(e));
                same(self, Type::Fn(ps, Box::new(r), effs), changed)
            }
            _ => {
                if !self.unify_exact(&ra, &rb) {
                    return None;
                }
                Some((a.clone(), false))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binds_and_resolves_variables() {
        let mut s = Subst::default();
        let a = s.fresh();
        assert!(s.unify(&Type::list(a.clone()), &Type::list(Type::Int)));
        assert_eq!(s.finish(&Type::option(a.clone())), Type::option(Type::Int));
        assert!(!s.unify(&a, &Type::Str));
        let b = s.fresh();
        assert_eq!(s.finish(&Type::list(b)), Type::list(Type::Unit));
    }

    #[test]
    fn rejects_infinite_types() {
        let mut s = Subst::default();
        let a = s.fresh();
        assert!(!s.unify(&a, &Type::list(a.clone())));
    }

    #[test]
    fn sizes_of_shared_variables_are_capped() {
        let mut s = Subst::default();
        let mut t = s.fresh();
        assert!(s.unify(&t, &Type::Int));
        // Each step builds Result[t, t]: the size doubles while the depth grows by one.
        for _ in 0..11 {
            let v = s.fresh();
            assert!(s.unify(&v, &Type::result(t.clone(), t.clone())));
            t = v;
        }
        let v = s.fresh();
        assert!(!s.unify(&v, &Type::result(t.clone(), t.clone())), "4095 * 2 + 1 nodes exceed 4096");
        assert!(s.too_large && !s.too_deep);
    }

    #[test]
    fn generic_type_arguments_are_invariant_in_effects() {
        let mut s = Subst::default();
        let pure = Type::Fn(vec![], Box::new(Type::Unit), vec![]);
        let io = Type::Fn(vec![], Box::new(Type::Unit), vec!["io".into()]);
        let app = |t: &Type| Type::App("Sink".into(), vec![t.clone()]);
        assert!(!s.unify(&app(&io), &app(&pure)));
        assert!(!s.unify(&app(&pure), &app(&io)));
        assert!(s.unify(&app(&io), &app(&io)));
        // Built-in containers stay covariant.
        assert!(s.unify(&Type::list(io.clone()), &Type::list(pure.clone())));
        let v = s.fresh();
        assert!(s.unify(&app(&v), &app(&Type::Int)));
        assert_eq!(s.finish(&v), Type::Int);
    }

    #[test]
    fn function_values_may_have_fewer_effects() {
        let mut s = Subst::default();
        let pure = Type::Fn(vec![Type::Int], Box::new(Type::Int), vec![]);
        let io = Type::Fn(vec![Type::Int], Box::new(Type::Int), vec!["io".into()]);
        assert!(s.unify(&io, &pure));
        assert!(!s.unify(&pure, &io));
    }
}

//! Type variables and unification for local type inference inside one definition.

use qal_core::ast::Type;

#[derive(Debug, Default)]
pub struct Subst {
    /// Binding of each inference variable `Type::Var(i)`.
    vars: Vec<Option<Type>>,
}

impl Subst {
    pub fn fresh(&mut self) -> Type {
        self.vars.push(None);
        Type::Var(self.vars.len() as u32 - 1)
    }

    /// Replaces bound variables (deeply). Unbound variables stay.
    pub fn resolve(&self, t: &Type) -> Type {
        match t {
            Type::Var(v) => match &self.vars[*v as usize] {
                Some(b) => self.resolve(b),
                None => t.clone(),
            },
            Type::List(x) => Type::list(self.resolve(x)),
            Type::Option(x) => Type::option(self.resolve(x)),
            Type::Result(a, b) => Type::result(self.resolve(a), self.resolve(b)),
            Type::Fn(ps, r, e) => Type::Fn(ps.iter().map(|p| self.resolve(p)).collect(), Box::new(self.resolve(r)), e.clone()),
            _ => t.clone(),
        }
    }

    /// Like `resolve`, but unbound variables become `Unit` (their representation never matters).
    pub fn finish(&self, t: &Type) -> Type {
        match self.resolve(t) {
            Type::Var(_) => Type::Unit,
            Type::List(x) => Type::list(self.finish(&x)),
            Type::Option(x) => Type::option(self.finish(&x)),
            Type::Result(a, b) => Type::result(self.finish(&a), self.finish(&b)),
            Type::Fn(ps, r, e) => Type::Fn(ps.iter().map(|p| self.finish(p)).collect(), Box::new(self.finish(&r)), e),
            t => t,
        }
    }

    fn occurs(&self, v: u32, t: &Type) -> bool {
        match self.resolve(t) {
            Type::Var(w) => v == w,
            Type::List(x) | Type::Option(x) => self.occurs(v, &x),
            Type::Result(a, b) => self.occurs(v, &a) || self.occurs(v, &b),
            Type::Fn(ps, r, _) => ps.iter().any(|p| self.occurs(v, p)) || self.occurs(v, &r),
            _ => false,
        }
    }

    /// Makes `expected` and `found` equal. A function value may have fewer effects than expected.
    /// On failure, bindings made so far are kept (the caller reports a diagnostic and stops).
    pub fn unify(&mut self, expected: &Type, found: &Type) -> bool {
        let (a, b) = (self.resolve(expected), self.resolve(found));
        match (&a, &b) {
            (Type::Var(x), Type::Var(y)) if x == y => true,
            (Type::Var(x), t) | (t, Type::Var(x)) => {
                if self.occurs(*x, t) {
                    return false;
                }
                self.vars[*x as usize] = Some(t.clone());
                true
            }
            (Type::List(x), Type::List(y)) | (Type::Option(x), Type::Option(y)) => self.unify(x, y),
            (Type::Result(a1, b1), Type::Result(a2, b2)) => self.unify(a1, a2) && self.unify(b1, b2),
            (Type::Fn(p1, r1, e1), Type::Fn(p2, r2, e2)) => {
                p1.len() == p2.len()
                    // Parameters are contravariant: effect subtyping flips.
                    && p1.iter().zip(p2).all(|(x, y)| self.unify(y, x))
                    && self.unify(r1, r2)
                    && e2.iter().all(|e| e1.contains(e))
            }
            _ => a == b,
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
        assert_eq!(s.resolve(&Type::option(a.clone())), Type::option(Type::Int));
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
    fn function_values_may_have_fewer_effects() {
        let mut s = Subst::default();
        let pure = Type::Fn(vec![Type::Int], Box::new(Type::Int), vec![]);
        let io = Type::Fn(vec![Type::Int], Box::new(Type::Int), vec!["io".into()]);
        assert!(s.unify(&io, &pure));
        assert!(!s.unify(&pure, &io));
    }
}

//! Typing of built-in function calls (spec §3.5).

use crate::infer::Cx;
use qal_core::ast::{Expr, Type};
use qal_core::diag::Diagnostic;
use qal_core::tast::{Builtin, TExpr, TKind};

impl Cx<'_> {
    /// Types a call of built-in `name`; None if `name` is not a built-in.
    pub fn builtin(&mut self, name: &str, args: &[Expr], path: &str) -> Option<TExpr> {
        use Builtin as B;
        let (a, b) = (self.sub.fresh(), self.sub.fresh());
        let fun = |ps: Vec<Type>, r: Type| Type::Fn(ps, Box::new(r), vec![]);
        let list = Type::list;
        let (builtin, params, ret) = match name {
            "len" => return Some(self.len(args, path)),
            "slice" => (B::Slice, vec![Type::Str, Type::Int, Type::Int], Type::Str),
            "split" => (B::Split, vec![Type::Str, Type::Str], list(Type::Str)),
            "join" => (B::Join, vec![list(Type::Str), Type::Str], Type::Str),
            "trim" => (B::Trim, vec![Type::Str], Type::Str),
            "contains" => (B::Contains, vec![Type::Str, Type::Str], Type::Bool),
            "starts_with" => (B::StartsWith, vec![Type::Str, Type::Str], Type::Bool),
            "to_int" => (B::ToInt, vec![Type::Str], Type::option(Type::Int)),
            "to_float" => (B::ToFloat, vec![Type::Str], Type::option(Type::Float)),
            "from_int" => (B::FromInt, vec![Type::Int], Type::Str),
            "from_float" => (B::FromFloat, vec![Type::Float], Type::Str),
            "chars" => (B::Chars, vec![Type::Str], list(Type::Str)),
            "get" => (B::Get, vec![list(a.clone()), Type::Int], Type::option(a)),
            "push" => (B::Push, vec![list(a.clone()), a.clone()], list(a)),
            "concat" => (B::Concat, vec![list(a.clone()), list(a.clone())], list(a)),
            "map" => (B::Map, vec![list(a.clone()), fun(vec![a], b.clone())], list(b)),
            "filter" => (B::Filter, vec![list(a.clone()), fun(vec![a.clone()], Type::Bool)], list(a)),
            "fold" => (B::Fold, vec![list(a.clone()), b.clone(), fun(vec![b.clone(), a], b.clone())], b),
            "range" => (B::Range, vec![Type::Int, Type::Int], list(Type::Int)),
            "reverse" => (B::Reverse, vec![list(a.clone())], list(a)),
            "sort_int" => (B::SortInt, vec![list(Type::Int)], list(Type::Int)),
            "int_to_float" => (B::IntToFloat, vec![Type::Int], Type::Float),
            "float_to_int" => (B::FloatToInt, vec![Type::Float], Type::option(Type::Int)),
            "print" => (B::Print, vec![Type::Str], Type::Unit),
            "read_line" => (B::ReadLine, vec![], Type::option(Type::Str)),
            "read_file" => (B::ReadFile, vec![Type::Str], Type::result(Type::Str, Type::Str)),
            "write_file" => (B::WriteFile, vec![Type::Str, Type::Str], Type::result(Type::Unit, Type::Str)),
            "now_ms" => (B::NowMs, vec![], Type::Int),
            "rand_int" => (B::RandInt, vec![Type::Int, Type::Int], Type::Int),
            _ => return None,
        };
        // Bound before the arguments bind `a`/`b`, so the result's depth stays tracked.
        let ret = self.sub.wrap(ret);
        if args.len() != params.len() {
            self.diags.push(Diagnostic::arity(self.def, path, name, params.len(), args.len()));
            self.infer_all(args, path);
            return Some(self.poison());
        }
        let mut targs = Vec::new();
        for (i, (arg, p)) in args.iter().zip(&params).enumerate() {
            let ap = format!("{path}.args[{i}]");
            targs.push(match p {
                Type::Fn(fps, fr, _) => self.fn_arg(name, arg, fps, fr, &ap),
                _ => self.expect(arg, p, &ap),
            });
        }
        if let Some(e) = builtin.effect() {
            self.effect_check(path, name, &[e.to_string()]);
        }
        Some(TExpr { ty: ret, kind: TKind::Builtin(builtin, targs, path.into()) })
    }

    /// A function argument of map/filter/fold: its effects become effects of the call.
    fn fn_arg(&mut self, callee: &str, arg: &Expr, params: &[Type], ret: &Type, path: &str) -> TExpr {
        let t = self.infer(arg, path);
        let effects = match self.sub.resolve(&t.ty) {
            Type::Fn(_, _, effs) => effs,
            _ => vec![],
        };
        let want = Type::Fn(params.to_vec(), Box::new(ret.clone()), effects.clone());
        if self.unify_at(path, &want, &t.ty) {
            self.effect_check(path, callee, &effects);
        }
        t
    }

    /// `len` works on Str and List; the variant is fixed once the argument type is known.
    fn len(&mut self, args: &[Expr], path: &str) -> TExpr {
        if args.len() != 1 {
            self.diags.push(Diagnostic::arity(self.def, path, "len", 1, args.len()));
            self.infer_all(args, path);
            return self.poison();
        }
        let ap = format!("{path}.args[0]");
        let x = self.infer(&args[0], &ap);
        let b = match self.sub.resolve(&x.ty) {
            Type::Str => Builtin::StrLen,
            Type::List(_) | Type::Var(_) => Builtin::ListLen,
            other => {
                let found = self.sub.finish(&other).to_string();
                self.diags.push(Diagnostic::type_mismatch(self.def, &ap, "Str or List[_]", &found));
                return self.poison();
            }
        };
        TExpr { ty: Type::Int, kind: TKind::Builtin(b, vec![x], path.into()) }
    }
}

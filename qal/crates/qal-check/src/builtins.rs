//! Typing of built-in function calls (spec §3.5).

use crate::infer::Cx;
use qal_core::ast::{Expr, Type};
use qal_core::diag::Diagnostic;
use qal_core::tast::{Builtin, TExpr, TKind};

impl Cx<'_> {
    /// Types a call of built-in `name`; None if `name` is not a built-in.
    pub fn builtin(&mut self, name: &str, args: &[Expr], path: &str) -> Option<TExpr> {
        if name == "len" {
            return Some(self.len(args, path));
        }
        let (a, b) = (self.sub.fresh(), self.sub.fresh());
        let (builtin, params, ret) = signature(name, a, b)?;
        if !self.g.prelude && matches!(builtin, Builtin::Http | Builtin::DbExec | Builtin::DbQuery | Builtin::AccList | Builtin::ModelAsk) {
            self.diags.push(Diagnostic::prelude_shadowed(self.def, path, name));
            self.infer_all(args, path);
            return Some(self.poison());
        }
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
        if matches!(builtin, Builtin::DbExec | Builtin::DbQuery) {
            self.sql_literal(name, &args[0], &format!("{path}.args[0]"));
        }
        Some(TExpr { ty: ret, kind: TKind::Builtin(builtin, targs, path.into()) })
    }

    /// The SQL text of `db_exec`/`db_query` is part of the program: a string literal that passes the
    /// static check (one statement, `?` parameters, no database or transaction control).
    fn sql_literal(&mut self, callee: &str, arg: &Expr, path: &str) {
        match arg {
            Expr::Str(sql) => {
                if let Err(reason) = qal_core::sql::check_statement(sql) {
                    self.diags.push(Diagnostic::sql_invalid(self.def, path, &reason));
                }
            }
            _ => self.diags.push(Diagnostic::sql_not_literal(self.def, path, callee)),
        }
    }

    /// A function argument of map/filter/fold: its effects become effects of the call.
    fn fn_arg(&mut self, callee: &str, arg: &Expr, params: &[Type], ret: &Type, path: &str) -> TExpr {
        let t = match arg {
            Expr::Lambda(ps, body) if ps.len() == params.len() => self.lambda(ps, body, path, Some(params)),
            _ => self.infer(arg, path),
        };
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

/// Number of parameters of built-in `name` (None if it is not one).
pub(crate) fn arity(name: &str) -> Option<usize> {
    if name == "len" {
        return Some(1);
    }
    signature(name, Type::Unit, Type::Unit).map(|(_, ps, _)| ps.len())
}

/// The built-in `name` (other than `len`), its parameter types and result type, with `a` and `b`
/// for its type variables.
fn signature(name: &str, a: Type, b: Type) -> Option<(Builtin, Vec<Type>, Type)> {
    use Builtin as B;
    let fun = |ps: Vec<Type>, r: Type| Type::Fn(ps, Box::new(r), vec![]);
    let list = Type::list;
    let named = |n: &str| Type::Named(n.to_string());
    Some(match name {
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
        "read_all" => (B::ReadAll, vec![], Type::Str),
        "sort_str" => (B::SortStr, vec![list(Type::Str)], list(Type::Str)),
        "char_code" => (B::CharCode, vec![Type::Str], Type::Int),
        "from_char_code" => (B::FromCharCode, vec![Type::Int], Type::Str),
        "to_upper" => (B::ToUpper, vec![Type::Str], Type::Str),
        "to_lower" => (B::ToLower, vec![Type::Str], Type::Str),
        "replace" => (B::Replace, vec![Type::Str, Type::Str, Type::Str], Type::Str),
        "index_of" => (B::IndexOf, vec![Type::Str, Type::Str], Type::option(Type::Int)),
        "map_new" => (B::MapNew, vec![], Type::map(a, b)),
        "map_get" => (B::MapGet, vec![Type::map(a.clone(), b.clone()), a], Type::option(b)),
        "map_put" => (B::MapPut, vec![Type::map(a.clone(), b.clone()), a.clone(), b.clone()], Type::map(a, b)),
        "map_del" => (B::MapDel, vec![Type::map(a.clone(), b.clone()), a.clone()], Type::map(a, b)),
        "map_has" => (B::MapHas, vec![Type::map(a.clone(), b), a], Type::Bool),
        "map_size" => (B::MapSize, vec![Type::map(a, b)], Type::Int),
        "map_keys" => (B::MapKeys, vec![Type::map(a.clone(), b)], list(a)),
        "map_values" => (B::MapValues, vec![Type::map(a, b.clone())], list(b)),
        "to_bytes" => (B::ToBytes, vec![Type::Str], Type::Bytes),
        "bytes_len" => (B::BytesLen, vec![Type::Bytes], Type::Int),
        "bytes_get" => (B::BytesGet, vec![Type::Bytes, Type::Int], Type::option(Type::Int)),
        "bytes_slice" => (B::BytesSlice, vec![Type::Bytes, Type::Int, Type::Int], Type::Bytes),
        "bytes_from" => (B::BytesFrom, vec![list(Type::Int)], Type::Bytes),
        "bytes_to_list" => (B::BytesToList, vec![Type::Bytes], list(Type::Int)),
        "bytes_to_str" => (B::BytesToStr, vec![Type::Bytes], Type::option(Type::Str)),
        "compare" => (B::Compare, vec![a.clone(), a], Type::Int),
        "http" => (B::Http, vec![named("HttpCall")], Type::result(named("HttpResponse"), named("NetError"))),
        "db_exec" => (B::DbExec, vec![Type::Str, list(named("SqlValue"))], Type::result(Type::Int, named("DbError"))),
        "db_query" => (B::DbQuery, vec![Type::Str, list(named("SqlValue"))], Type::result(named("DbRows"), named("DbError"))),
        "model_ask" => (B::ModelAsk, vec![Type::Str, Type::Str], Type::result(named("ModelResponse"), named("ModelError"))),
        // Phase 5a (self-hosting spec §5.2): pure, allocation-free or linear building blocks.
        "bit_and" => (B::BitAnd, vec![Type::Int, Type::Int], Type::Int),
        "bit_or" => (B::BitOr, vec![Type::Int, Type::Int], Type::Int),
        "bit_xor" => (B::BitXor, vec![Type::Int, Type::Int], Type::Int),
        "bit_not" => (B::BitNot, vec![Type::Int], Type::Int),
        "shl" => (B::Shl, vec![Type::Int, Type::Int], Type::Int),
        "shr" => (B::Shr, vec![Type::Int, Type::Int], Type::Int),
        "ushr" => (B::Ushr, vec![Type::Int, Type::Int], Type::Int),
        "float_bits" => (B::FloatBits, vec![Type::Float], Type::Int),
        "float_of_bits" => (B::FloatOfBits, vec![Type::Int], Type::Float),
        "list_at" => (B::ListAt, vec![list(a.clone()), Type::Int], a),
        "bytes_at" => (B::BytesAt, vec![Type::Bytes, Type::Int], Type::Int),
        "acc_list" => (B::AccList, vec![Type::App(qal_core::prelude::ACCUM.into(), vec![a.clone()])], list(a)),
        "bytes_join" => (B::BytesJoin, vec![list(Type::Bytes)], Type::Bytes),
        "sha256" => (B::Sha256, vec![Type::Bytes], Type::Bytes),
        // A stub (Phase 4 §11): any type, from the expected type; traps with `todo_reached` (E0540).
        "todo" => (B::Todo, vec![], a),
        _ => return None,
    })
}

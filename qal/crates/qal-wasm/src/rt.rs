//! Runtime library: allocator, checked arithmetic and the built-in Str/List functions, written
//! directly in WebAssembly. Each helper is generated once, on first use.
//!
//! Memory layout (all blocks 8-byte aligned, see `crate` docs):
//!   Str  = [len: i32][UTF-8 bytes]          List = [len: i32][pad][slot; len]
//!   variant = [tag: i32][pad][slot; fields]  record = [slot; fields]
//! A slot is 8 bytes: i64 as is, f64 bit pattern, i32 zero-extended.

use crate::asm::{from_slot, m32, m64, m8, to_slot, Asm, Repr};
use crate::module::{Arg, Mb};
use crate::{host, FAIL_BUILTIN, FAIL_DIV_ZERO, FAIL_INDEX, FAIL_OOM, FAIL_OVERFLOW, HEAP_GLOBAL, OOM_SCRATCH};
use qal_core::ast::BinOp;
use qal_core::tast::{TAG_ERR, TAG_NONE, TAG_OK, TAG_SOME};
use wasm_encoder::{BlockType, Function, Instruction as I, ValType};

use ValType::{F64, I32, I64};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rt {
    /// (size: i64) -> ptr; fails with out_of_memory beyond the memory limit.
    Alloc,
    /// Checked Int `+ - * / %`: (a: i64, b: i64, func: i32, site: i32) -> i64.
    Arith(BinOp),
    /// (tag: i32, slot: i64) -> ptr of a one-field variant.
    Variant1,
    /// (ptr or -1) -> Option[Str]
    OptFromPtr,
    /// (ok: i32, str: i32) -> Result[Str, Str]
    ResultFrom,
    /// (ok: i32, err: i32) -> Result[Unit, Str]
    ResultUnitFrom,
    /// (len: i32) -> fresh Str with uninitialized bytes
    StrNew,
    /// (s) -> number of chars (i64)
    StrLen,
    StrConcat,
    /// (a_ptr, b_ptr, n) -> 1 if the n bytes are equal
    BytesEq,
    StrEq,
    /// (s, from_byte, to_byte) -> new Str
    StrSub,
    /// (s, char_index: i64) -> byte offset of that char (or byte length)
    CharToByte,
    /// (s, start: i64, end: i64, func, site) -> Str; fails with `index`
    StrSlice,
    /// (s, sub, from_byte) -> byte offset or -1
    StrFind,
    StrContains,
    StrStartsWith,
    /// (s, sep, func, site) -> List[Str]; fails if sep is empty
    StrSplit,
    StrJoin,
    StrTrim,
    /// (s) -> Option[Int]
    StrToInt,
    /// (n: i64) -> Str
    IntToStr,
    StrChars,
    /// (n: i32) -> fresh List with uninitialized slots
    ListNew,
    /// (xs, i: i64) -> Option[slot]
    ListGet,
    /// (xs, slot: i64) -> List
    ListPush,
    ListConcat,
    ListReverse,
    /// (a: i64, b: i64) -> List[Int]
    ListRange,
    ListSortInt,
    /// (base, root, end): heap sift-down over i64 elements
    Sift,
    /// (x: f64) -> Option[Int]
    FloatToInt,
    /// (lo: i64, hi: i64, func, site) -> i64
    RandChecked,
    /// (xs, f) -> List; f: (in) -> out
    Map(Repr, Repr),
    /// (xs, f) -> List; f: (in) -> Bool
    Filter(Repr),
    /// (xs, init: acc, f) -> acc; f: (acc, elem) -> acc
    Fold(Repr, Repr),
}

/// Pushes the address of slot `k` of list `xs` (use with offset 8).
fn slot_addr(a: &mut Asm, xs: u32, k: u32) {
    a.all([I::LocalGet(xs), I::LocalGet(k), I::I32Const(3), I::I32Shl, I::I32Add]);
}

/// `block { loop { <cond> br_if 1 ... } }` opener: exits when `k >= n` (unsigned i32).
fn loop_until(a: &mut Asm, k: u32, n: u32) {
    a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty), I::LocalGet(k), I::LocalGet(n), I::I32GeU, I::BrIf(1)]);
}

/// Closes a `loop_until` loop, incrementing `k` by one.
fn loop_next(a: &mut Asm, k: u32) {
    a.all([I::LocalGet(k), I::I32Const(1), I::I32Add, I::LocalSet(k), I::Br(0), I::End, I::End]);
}

fn incr(a: &mut Asm, l: u32) {
    a.all([I::LocalGet(l), I::I32Const(1), I::I32Add, I::LocalSet(l)]);
}

impl Rt {
    pub fn signature(self) -> (Vec<ValType>, Vec<ValType>) {
        let (p, r): (&[ValType], &[ValType]) = match self {
            Rt::Alloc => (&[I64], &[I32]),
            Rt::Arith(_) => (&[I64, I64, I32, I32], &[I64]),
            Rt::Variant1 => (&[I32, I64], &[I32]),
            Rt::OptFromPtr | Rt::StrNew | Rt::StrTrim | Rt::StrToInt | Rt::StrChars | Rt::ListNew | Rt::ListReverse | Rt::ListSortInt => (&[I32], &[I32]),
            Rt::ResultFrom | Rt::ResultUnitFrom | Rt::StrConcat | Rt::StrEq | Rt::StrContains | Rt::StrStartsWith | Rt::StrJoin | Rt::ListConcat => (&[I32, I32], &[I32]),
            Rt::StrLen => (&[I32], &[I64]),
            Rt::BytesEq | Rt::StrSub | Rt::StrFind => (&[I32, I32, I32], &[I32]),
            Rt::CharToByte => (&[I32, I64], &[I32]),
            Rt::StrSlice => (&[I32, I64, I64, I32, I32], &[I32]),
            Rt::StrSplit => (&[I32, I32, I32, I32], &[I32]),
            Rt::IntToStr => (&[I64], &[I32]),
            Rt::ListGet | Rt::ListPush => (&[I32, I64], &[I32]),
            Rt::ListRange => (&[I64, I64], &[I32]),
            Rt::Sift => (&[I32, I32, I32], &[]),
            Rt::FloatToInt => (&[F64], &[I32]),
            Rt::RandChecked => (&[I64, I64, I32, I32], &[I64]),
            Rt::Map(..) | Rt::Filter(_) => (&[I32, I32], &[I32]),
            Rt::Fold(acc, _) => return (vec![I32, acc.val(), I32], vec![acc.val()]),
        };
        (p.to_vec(), r.to_vec())
    }

    pub fn body(self, m: &mut Mb, n_params: usize) -> Function {
        let mut a = Asm::new(n_params);
        match self {
            Rt::Alloc => alloc(m, &mut a),
            Rt::Arith(op) => arith(m, &mut a, op),
            Rt::Variant1 => {
                let p = a.local(I32);
                a.all([I::I64Const(16), I::Call(m.rt(Rt::Alloc)), I::LocalTee(p), I::LocalGet(0), I::I32Store(m32(0))]);
                a.all([I::LocalGet(p), I::LocalGet(1), I::I64Store(m64(8)), I::LocalGet(p)]);
            }
            Rt::OptFromPtr => {
                a.all([I::LocalGet(0), I::I32Const(0), I::I32LtS, I::If(BlockType::Result(I32)), I::I32Const(m.tag_addr(TAG_NONE)), I::Else]);
                a.all([I::I32Const(TAG_SOME as i32), I::LocalGet(0), I::I64ExtendI32U, I::Call(m.rt(Rt::Variant1)), I::End]);
            }
            Rt::ResultFrom | Rt::ResultUnitFrom => {
                // ok ? Ok(x) : Err(str); for Result[Unit, Str] the Ok payload is Unit (0).
                a.all([I::I32Const(TAG_OK as i32), I::I32Const(TAG_ERR as i32), I::LocalGet(0), I::Select]);
                if self == Rt::ResultFrom {
                    a.i(I::LocalGet(1));
                } else {
                    a.all([I::I32Const(0), I::LocalGet(1), I::LocalGet(0), I::Select]);
                }
                a.all([I::I64ExtendI32U, I::Call(m.rt(Rt::Variant1))]);
            }
            Rt::StrNew => {
                let p = a.local(I32);
                a.all([I::LocalGet(0), I::I64ExtendI32U, I::I64Const(4), I::I64Add, I::Call(m.rt(Rt::Alloc)), I::LocalTee(p)]);
                a.all([I::LocalGet(0), I::I32Store(m32(0)), I::LocalGet(p)]);
            }
            Rt::StrLen => {
                let (n, i, c) = (a.local(I32), a.local(I32), a.local(I64));
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalSet(n)]);
                loop_until(&mut a, i, n);
                // count bytes that are not UTF-8 continuation bytes (10xxxxxx)
                a.all([I::LocalGet(0), I::LocalGet(i), I::I32Add, I::I32Load8U(m8(4)), I::I32Const(0xC0), I::I32And, I::I32Const(0x80), I::I32Ne]);
                a.all([I::I64ExtendI32U, I::LocalGet(c), I::I64Add, I::LocalSet(c)]);
                loop_next(&mut a, i);
                a.i(I::LocalGet(c));
            }
            Rt::StrConcat => {
                let (la, lb, p) = (a.local(I32), a.local(I32), a.local(I32));
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalSet(la), I::LocalGet(1), I::I32Load(m32(0)), I::LocalSet(lb)]);
                a.all([I::LocalGet(la), I::LocalGet(lb), I::I32Add, I::Call(m.rt(Rt::StrNew)), I::LocalSet(p)]);
                a.all([I::LocalGet(p), I::I32Const(4), I::I32Add, I::LocalGet(0), I::I32Const(4), I::I32Add, I::LocalGet(la), copy()]);
                a.all([I::LocalGet(p), I::I32Const(4), I::I32Add, I::LocalGet(la), I::I32Add, I::LocalGet(1), I::I32Const(4), I::I32Add, I::LocalGet(lb), copy()]);
                a.i(I::LocalGet(p));
            }
            Rt::BytesEq => {
                let i = a.local(I32);
                loop_until(&mut a, i, 2);
                a.all([I::LocalGet(0), I::LocalGet(i), I::I32Add, I::I32Load8U(m8(0)), I::LocalGet(1), I::LocalGet(i), I::I32Add, I::I32Load8U(m8(0)), I::I32Ne]);
                a.all([I::If(BlockType::Empty), I::I32Const(0), I::Return, I::End]);
                loop_next(&mut a, i);
                a.i(I::I32Const(1));
            }
            Rt::StrEq => {
                a.all([I::LocalGet(0), I::LocalGet(1), I::I32Eq, I::If(BlockType::Empty), I::I32Const(1), I::Return, I::End]);
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalGet(1), I::I32Load(m32(0)), I::I32Ne, I::If(BlockType::Empty), I::I32Const(0), I::Return, I::End]);
                a.all([I::LocalGet(0), I::I32Const(4), I::I32Add, I::LocalGet(1), I::I32Const(4), I::I32Add, I::LocalGet(0), I::I32Load(m32(0)), I::Call(m.rt(Rt::BytesEq))]);
            }
            Rt::StrSub => {
                let p = a.local(I32);
                a.all([I::LocalGet(2), I::LocalGet(1), I::I32Sub, I::Call(m.rt(Rt::StrNew)), I::LocalSet(p)]);
                a.all([I::LocalGet(p), I::I32Const(4), I::I32Add, I::LocalGet(0), I::I32Const(4), I::I32Add, I::LocalGet(1), I::I32Add, I::LocalGet(2), I::LocalGet(1), I::I32Sub, copy()]);
                a.i(I::LocalGet(p));
            }
            Rt::CharToByte => {
                let (n, i, c) = (a.local(I32), a.local(I32), a.local(I64));
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalSet(n)]);
                loop_until(&mut a, i, n);
                a.all([I::LocalGet(0), I::LocalGet(i), I::I32Add, I::I32Load8U(m8(4)), I::I32Const(0xC0), I::I32And, I::I32Const(0x80), I::I32Ne, I::If(BlockType::Empty)]);
                a.all([I::LocalGet(c), I::LocalGet(1), I::I64Eq, I::If(BlockType::Empty), I::LocalGet(i), I::Return, I::End]);
                a.all([I::LocalGet(c), I::I64Const(1), I::I64Add, I::LocalSet(c), I::End]);
                loop_next(&mut a, i);
                a.i(I::LocalGet(n));
            }
            Rt::StrSlice => {
                let n = a.local(I64);
                a.all([I::LocalGet(0), I::Call(m.rt(Rt::StrLen)), I::LocalSet(n)]);
                a.all([I::LocalGet(1), I::I64Const(0), I::I64LtS, I::LocalGet(2), I::LocalGet(1), I::I64LtS, I::I32Or, I::LocalGet(2), I::LocalGet(n), I::I64GtS, I::I32Or]);
                a.i(I::If(BlockType::Empty));
                m.emit_fail(&mut a, FAIL_INDEX, Arg::Local(3), Arg::Local(4), &[(1, Repr::I64), (2, Repr::I64), (n, Repr::I64)]);
                a.i(I::End);
                let c2b = m.rt(Rt::CharToByte);
                a.all([I::LocalGet(0), I::LocalGet(0), I::LocalGet(1), I::Call(c2b), I::LocalGet(0), I::LocalGet(2), I::Call(c2b), I::Call(m.rt(Rt::StrSub))]);
            }
            Rt::StrFind => {
                let (ls, lsub, i) = (a.local(I32), a.local(I32), a.local(I32));
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalSet(ls), I::LocalGet(1), I::I32Load(m32(0)), I::LocalSet(lsub), I::LocalGet(2), I::LocalSet(i)]);
                a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty)]);
                a.all([I::LocalGet(i), I::LocalGet(lsub), I::I32Add, I::LocalGet(ls), I::I32GtU, I::BrIf(1)]);
                a.all([I::LocalGet(0), I::I32Const(4), I::I32Add, I::LocalGet(i), I::I32Add, I::LocalGet(1), I::I32Const(4), I::I32Add, I::LocalGet(lsub), I::Call(m.rt(Rt::BytesEq))]);
                a.all([I::If(BlockType::Empty), I::LocalGet(i), I::Return, I::End]);
                incr(&mut a, i);
                a.all([I::Br(0), I::End, I::End, I::I32Const(-1)]);
            }
            Rt::StrContains => {
                a.all([I::LocalGet(0), I::LocalGet(1), I::I32Const(0), I::Call(m.rt(Rt::StrFind)), I::I32Const(-1), I::I32Ne]);
            }
            Rt::StrStartsWith => {
                a.all([I::LocalGet(1), I::I32Load(m32(0)), I::LocalGet(0), I::I32Load(m32(0)), I::I32GtU, I::If(BlockType::Empty), I::I32Const(0), I::Return, I::End]);
                a.all([I::LocalGet(0), I::I32Const(4), I::I32Add, I::LocalGet(1), I::I32Const(4), I::I32Add, I::LocalGet(1), I::I32Load(m32(0)), I::Call(m.rt(Rt::BytesEq))]);
            }
            Rt::StrSplit => split(m, &mut a),
            Rt::StrJoin => join(m, &mut a),
            Rt::StrTrim => {
                let (s, e, c) = (a.local(I32), a.local(I32), a.local(I32));
                let is_ws = |a: &mut Asm| {
                    a.all([I::LocalTee(c), I::I32Const(32), I::I32Eq, I::LocalGet(c), I::I32Const(9), I::I32Eq, I::I32Or]);
                    a.all([I::LocalGet(c), I::I32Const(10), I::I32Eq, I::I32Or, I::LocalGet(c), I::I32Const(13), I::I32Eq, I::I32Or]);
                };
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalSet(e)]);
                a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty), I::LocalGet(s), I::LocalGet(e), I::I32GeU, I::BrIf(1)]);
                a.all([I::LocalGet(0), I::LocalGet(s), I::I32Add, I::I32Load8U(m8(4))]);
                is_ws(&mut a);
                a.all([I::I32Eqz, I::BrIf(1)]);
                incr(&mut a, s);
                a.all([I::Br(0), I::End, I::End]);
                a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty), I::LocalGet(e), I::LocalGet(s), I::I32LeU, I::BrIf(1)]);
                a.all([I::LocalGet(0), I::LocalGet(e), I::I32Add, I::I32Load8U(m8(3))]);
                is_ws(&mut a);
                a.all([I::I32Eqz, I::BrIf(1), I::LocalGet(e), I::I32Const(1), I::I32Sub, I::LocalSet(e), I::Br(0), I::End, I::End]);
                a.all([I::LocalGet(0), I::LocalGet(s), I::LocalGet(e), I::Call(m.rt(Rt::StrSub))]);
            }
            Rt::StrToInt => str_to_int(m, &mut a),
            Rt::IntToStr => int_to_str(m, &mut a),
            Rt::StrChars => {
                let (xs, nb, i, k, cl) = (a.local(I32), a.local(I32), a.local(I32), a.local(I32), a.local(I32));
                a.all([I::LocalGet(0), I::Call(m.rt(Rt::StrLen)), I::I32WrapI64, I::Call(m.rt(Rt::ListNew)), I::LocalSet(xs)]);
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalSet(nb)]);
                loop_until(&mut a, i, nb);
                // UTF-8 sequence length from the lead byte: 1 + (b >= 0xC0) + (b >= 0xE0) + (b >= 0xF0)
                let lead = [I::LocalGet(0), I::LocalGet(i), I::I32Add, I::I32Load8U(m8(4))];
                a.i(I::I32Const(1));
                for bound in [0xC0, 0xE0, 0xF0] {
                    a.all(lead.clone());
                    a.all([I::I32Const(bound), I::I32GeU, I::I32Add]);
                }
                a.i(I::LocalSet(cl));
                slot_addr(&mut a, xs, k);
                a.all([I::LocalGet(0), I::LocalGet(i), I::LocalGet(i), I::LocalGet(cl), I::I32Add, I::Call(m.rt(Rt::StrSub)), I::I64ExtendI32U, I::I64Store(m64(8))]);
                incr(&mut a, k);
                a.all([I::LocalGet(i), I::LocalGet(cl), I::I32Add, I::LocalSet(i), I::Br(0), I::End, I::End]);
                a.i(I::LocalGet(xs));
            }
            Rt::ListNew => {
                let p = a.local(I32);
                a.all([I::LocalGet(0), I::I64ExtendI32U, I::I64Const(3), I::I64Shl, I::I64Const(8), I::I64Add, I::Call(m.rt(Rt::Alloc)), I::LocalTee(p)]);
                a.all([I::LocalGet(0), I::I32Store(m32(0)), I::LocalGet(p)]);
            }
            Rt::ListGet => {
                a.all([I::LocalGet(1), I::I64Const(0), I::I64LtS, I::LocalGet(1), I::LocalGet(0), I::I32Load(m32(0)), I::I64ExtendI32U, I::I64GeS, I::I32Or]);
                a.all([I::If(BlockType::Result(I32)), I::I32Const(m.tag_addr(TAG_NONE)), I::Else]);
                a.all([I::I32Const(TAG_SOME as i32), I::LocalGet(0), I::LocalGet(1), I::I32WrapI64, I::I32Const(3), I::I32Shl, I::I32Add, I::I64Load(m64(8))]);
                a.all([I::Call(m.rt(Rt::Variant1)), I::End]);
            }
            Rt::ListPush => {
                let (n, p) = (a.local(I32), a.local(I32));
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalTee(n), I::I32Const(1), I::I32Add, I::Call(m.rt(Rt::ListNew)), I::LocalSet(p)]);
                a.all([I::LocalGet(p), I::I32Const(8), I::I32Add, I::LocalGet(0), I::I32Const(8), I::I32Add, I::LocalGet(n), I::I32Const(3), I::I32Shl, copy()]);
                slot_addr(&mut a, p, n);
                a.all([I::LocalGet(1), I::I64Store(m64(8)), I::LocalGet(p)]);
            }
            Rt::ListConcat => {
                let (na, nb, p) = (a.local(I32), a.local(I32), a.local(I32));
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalSet(na), I::LocalGet(1), I::I32Load(m32(0)), I::LocalSet(nb)]);
                a.all([I::LocalGet(na), I::LocalGet(nb), I::I32Add, I::Call(m.rt(Rt::ListNew)), I::LocalSet(p)]);
                a.all([I::LocalGet(p), I::I32Const(8), I::I32Add, I::LocalGet(0), I::I32Const(8), I::I32Add, I::LocalGet(na), I::I32Const(3), I::I32Shl, copy()]);
                slot_addr(&mut a, p, na);
                a.all([I::I32Const(8), I::I32Add, I::LocalGet(1), I::I32Const(8), I::I32Add, I::LocalGet(nb), I::I32Const(3), I::I32Shl, copy(), I::LocalGet(p)]);
            }
            Rt::ListReverse => {
                let (n, p, k) = (a.local(I32), a.local(I32), a.local(I32));
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalTee(n), I::Call(m.rt(Rt::ListNew)), I::LocalSet(p)]);
                loop_until(&mut a, k, n);
                slot_addr(&mut a, p, k);
                a.all([I::LocalGet(0), I::LocalGet(n), I::LocalGet(k), I::I32Sub, I::I32Const(3), I::I32Shl, I::I32Add, I::I64Load(m64(0)), I::I64Store(m64(8))]);
                loop_next(&mut a, k);
                a.i(I::LocalGet(p));
            }
            Rt::ListRange => {
                let (count, p, k) = (a.local(I64), a.local(I32), a.local(I32));
                a.all([I::LocalGet(1), I::LocalGet(0), I::I64LeS, I::If(BlockType::Empty), I::I32Const(0), I::Call(m.rt(Rt::ListNew)), I::Return, I::End]);
                // b - a may exceed i64::MAX; as unsigned it is exact. Absurd sizes are clamped so alloc reports out_of_memory.
                a.all([I::LocalGet(1), I::LocalGet(0), I::I64Sub, I::LocalSet(count)]);
                a.all([I::I64Const(1 << 40), I::LocalGet(count), I::I64Const(3), I::I64Shl, I::I64Const(8), I::I64Add, I::LocalGet(count), I::I64Const(1 << 32), I::I64GtU, I::Select]);
                a.all([I::Call(m.rt(Rt::Alloc)), I::LocalTee(p), I::LocalGet(count), I::I32WrapI64, I::I32Store(m32(0))]);
                let n = a.local(I32);
                a.all([I::LocalGet(count), I::I32WrapI64, I::LocalSet(n)]);
                loop_until(&mut a, k, n);
                slot_addr(&mut a, p, k);
                a.all([I::LocalGet(0), I::LocalGet(k), I::I64ExtendI32U, I::I64Add, I::I64Store(m64(8))]);
                loop_next(&mut a, k);
                a.i(I::LocalGet(p));
            }
            Rt::ListSortInt => sort_int(m, &mut a),
            Rt::Sift => sift(&mut a),
            Rt::FloatToInt => {
                a.all([I::LocalGet(0), I::LocalGet(0), I::F64Ne, I::LocalGet(0), I::F64Const(9223372036854775808.0.into()), I::F64Ge, I::I32Or]);
                a.all([I::LocalGet(0), I::F64Const((-9223372036854775808.0).into()), I::F64Lt, I::I32Or]);
                a.all([I::If(BlockType::Result(I32)), I::I32Const(m.tag_addr(TAG_NONE)), I::Else]);
                a.all([I::I32Const(TAG_SOME as i32), I::LocalGet(0), I::I64TruncF64S, I::Call(m.rt(Rt::Variant1)), I::End]);
            }
            Rt::RandChecked => {
                a.all([I::LocalGet(0), I::LocalGet(1), I::I64GtS, I::If(BlockType::Empty)]);
                m.emit_fail(&mut a, FAIL_BUILTIN, Arg::Local(2), Arg::Local(3), &[(0, Repr::I64), (1, Repr::I64)]);
                a.all([I::End, I::LocalGet(0), I::LocalGet(1), I::Call(m.import(host::RAND_INT))]);
            }
            Rt::Map(from, to) => {
                let (n, p, k) = (a.local(I32), a.local(I32), a.local(I32));
                let ty = m.ty(&[from.val()], &[to.val()]);
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalTee(n), I::Call(m.rt(Rt::ListNew)), I::LocalSet(p)]);
                loop_until(&mut a, k, n);
                slot_addr(&mut a, p, k);
                slot_addr(&mut a, 0, k);
                a.i(I::I64Load(m64(8)));
                a.opt(from_slot(from));
                a.all([I::LocalGet(1), I::CallIndirect { type_index: ty, table_index: 0 }]);
                a.opt(to_slot(to));
                a.i(I::I64Store(m64(8)));
                loop_next(&mut a, k);
                a.i(I::LocalGet(p));
            }
            Rt::Filter(elem) => {
                let (n, p, k, kept, v) = (a.local(I32), a.local(I32), a.local(I32), a.local(I32), a.local(I64));
                let ty = m.ty(&[elem.val()], &[I32]);
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalTee(n), I::Call(m.rt(Rt::ListNew)), I::LocalSet(p)]);
                loop_until(&mut a, k, n);
                slot_addr(&mut a, 0, k);
                a.all([I::I64Load(m64(8)), I::LocalTee(v)]);
                a.opt(from_slot(elem));
                a.all([I::LocalGet(1), I::CallIndirect { type_index: ty, table_index: 0 }, I::If(BlockType::Empty)]);
                slot_addr(&mut a, p, kept);
                a.all([I::LocalGet(v), I::I64Store(m64(8))]);
                incr(&mut a, kept);
                a.i(I::End);
                loop_next(&mut a, k);
                // The fresh list was sized for all elements; shrink its length to the kept ones.
                a.all([I::LocalGet(p), I::LocalGet(kept), I::I32Store(m32(0)), I::LocalGet(p)]);
            }
            Rt::Fold(acc, elem) => {
                let (n, k) = (a.local(I32), a.local(I32));
                let ty = m.ty(&[acc.val(), elem.val()], &[acc.val()]);
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalSet(n)]);
                loop_until(&mut a, k, n);
                a.i(I::LocalGet(1));
                slot_addr(&mut a, 0, k);
                a.i(I::I64Load(m64(8)));
                a.opt(from_slot(elem));
                a.all([I::LocalGet(2), I::CallIndirect { type_index: ty, table_index: 0 }, I::LocalSet(1)]);
                loop_next(&mut a, k);
                a.i(I::LocalGet(1));
            }
        }
        a.finish()
    }
}

fn copy() -> I<'static> {
    I::MemoryCopy { src_mem: 0, dst_mem: 0 }
}

/// Bump allocator over the exported memory; grows memory page-wise up to its maximum.
fn alloc(m: &mut Mb, a: &mut Asm) {
    let (p, end, have) = (a.local(I32), a.local(I64), a.local(I64));
    a.all([I::GlobalGet(HEAP_GLOBAL), I::LocalTee(p), I::I64ExtendI32U, I::LocalGet(0), I::I64Add, I::I64Const(7), I::I64Add, I::I64Const(-8), I::I64And, I::LocalSet(end)]);
    a.all([I::MemorySize(0), I::I64ExtendI32U, I::I64Const(16), I::I64Shl, I::LocalSet(have)]);
    a.all([I::LocalGet(end), I::LocalGet(have), I::I64GtU, I::If(BlockType::Empty)]);
    // pages needed = ceil((end - have) / 64 KiB); more than 4 GiB can never be granted
    a.all([I::LocalGet(end), I::I64Const(0xFFFF_FFFF), I::I64GtU, I::If(BlockType::Result(I32)), I::I32Const(-1), I::Else]);
    a.all([I::LocalGet(end), I::LocalGet(have), I::I64Sub, I::I64Const(65535), I::I64Add, I::I64Const(16), I::I64ShrU, I::I32WrapI64, I::MemoryGrow(0), I::End]);
    a.all([I::I32Const(-1), I::I32Eq, I::If(BlockType::Empty)]);
    a.all([I::I32Const(OOM_SCRATCH as i32), I::LocalGet(0), I::I64Store(m64(0))]);
    a.all([I::I32Const(FAIL_OOM), I::I32Const(-1), I::I32Const(-1), I::I32Const(OOM_SCRATCH as i32), I::Call(m.import(host::FAIL)), I::Unreachable]);
    a.all([I::End, I::End]);
    a.all([I::LocalGet(end), I::I32WrapI64, I::GlobalSet(HEAP_GLOBAL), I::LocalGet(p)]);
}

/// Checked Int arithmetic. Params: a = 0, b = 1, func = 2, site = 3; local r = 4.
fn arith(m: &mut Mb, a: &mut Asm, op: BinOp) {
    let r = a.local(I64);
    let vals = [(0, Repr::I64), (1, Repr::I64)];
    let fail_if = |m: &mut Mb, a: &mut Asm, kind: i32| {
        a.i(I::If(BlockType::Empty));
        m.emit_fail(a, kind, Arg::Local(2), Arg::Local(3), &vals);
        a.i(I::End);
    };
    match op {
        BinOp::Add | BinOp::Sub => {
            a.all([I::LocalGet(0), I::LocalGet(1), if op == BinOp::Add { I::I64Add } else { I::I64Sub }, I::LocalSet(r)]);
            // add overflows iff ((a ^ r) & (b ^ r)) < 0; sub iff ((a ^ b) & (a ^ r)) < 0
            if op == BinOp::Add {
                a.all([I::LocalGet(0), I::LocalGet(r), I::I64Xor, I::LocalGet(1), I::LocalGet(r), I::I64Xor]);
            } else {
                a.all([I::LocalGet(0), I::LocalGet(1), I::I64Xor, I::LocalGet(0), I::LocalGet(r), I::I64Xor]);
            }
            a.all([I::I64And, I::I64Const(0), I::I64LtS]);
            fail_if(m, a, FAIL_OVERFLOW);
            a.i(I::LocalGet(r));
        }
        BinOp::Mul => {
            // a == -1 && b == MIN overflows (and would trap in the division check below)
            a.all([I::LocalGet(0), I::I64Const(-1), I::I64Eq, I::LocalGet(1), I::I64Const(i64::MIN), I::I64Eq, I::I32And]);
            fail_if(m, a, FAIL_OVERFLOW);
            a.all([I::LocalGet(0), I::LocalGet(1), I::I64Mul, I::LocalSet(r)]);
            // a != 0 && r / a != b
            a.all([I::LocalGet(0), I::I64Const(0), I::I64Ne, I::If(BlockType::Empty)]);
            a.all([I::LocalGet(r), I::LocalGet(0), I::I64DivS, I::LocalGet(1), I::I64Ne]);
            fail_if(m, a, FAIL_OVERFLOW);
            a.all([I::End, I::LocalGet(r)]);
        }
        BinOp::Div | BinOp::Rem => {
            a.all([I::LocalGet(1), I::I64Eqz]);
            fail_if(m, a, FAIL_DIV_ZERO);
            if op == BinOp::Div {
                a.all([I::LocalGet(0), I::I64Const(i64::MIN), I::I64Eq, I::LocalGet(1), I::I64Const(-1), I::I64Eq, I::I32And]);
                fail_if(m, a, FAIL_OVERFLOW);
                a.all([I::LocalGet(0), I::LocalGet(1), I::I64DivS]);
            } else {
                // wasm i64.rem_s(MIN, -1) is defined as 0, no trap
                a.all([I::LocalGet(0), I::LocalGet(1), I::I64RemS]);
            }
        }
        _ => m.bug(format!("not an arithmetic operator: {op:?}")),
    }
}

/// split(s, sep). Params: s = 0, sep = 1, func = 2, site = 3.
fn split(m: &mut Mb, a: &mut Asm) {
    let (lsep, count, pos, f, xs, k) = (a.local(I32), a.local(I32), a.local(I32), a.local(I32), a.local(I32), a.local(I32));
    let find = m.rt(Rt::StrFind);
    let sub = m.rt(Rt::StrSub);
    a.all([I::LocalGet(1), I::I32Load(m32(0)), I::LocalTee(lsep), I::I32Eqz, I::If(BlockType::Empty)]);
    m.emit_fail(a, FAIL_BUILTIN, Arg::Local(2), Arg::Local(3), &[(1, Repr::I32)]);
    a.i(I::End);
    // pass 1: count parts
    a.all([I::I32Const(1), I::LocalSet(count)]);
    a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty)]);
    a.all([I::LocalGet(0), I::LocalGet(1), I::LocalGet(pos), I::Call(find), I::LocalTee(f), I::I32Const(0), I::I32LtS, I::BrIf(1)]);
    incr(a, count);
    a.all([I::LocalGet(f), I::LocalGet(lsep), I::I32Add, I::LocalSet(pos), I::Br(0), I::End, I::End]);
    // pass 2: cut
    a.all([I::LocalGet(count), I::Call(m.rt(Rt::ListNew)), I::LocalSet(xs), I::I32Const(0), I::LocalSet(pos)]);
    a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty)]);
    a.all([I::LocalGet(0), I::LocalGet(1), I::LocalGet(pos), I::Call(find), I::LocalTee(f), I::I32Const(0), I::I32LtS, I::BrIf(1)]);
    slot_addr(a, xs, k);
    a.all([I::LocalGet(0), I::LocalGet(pos), I::LocalGet(f), I::Call(sub), I::I64ExtendI32U, I::I64Store(m64(8))]);
    incr(a, k);
    a.all([I::LocalGet(f), I::LocalGet(lsep), I::I32Add, I::LocalSet(pos), I::Br(0), I::End, I::End]);
    slot_addr(a, xs, k);
    a.all([I::LocalGet(0), I::LocalGet(pos), I::LocalGet(0), I::I32Load(m32(0)), I::Call(sub), I::I64ExtendI32U, I::I64Store(m64(8)), I::LocalGet(xs)]);
}

/// join(xs, sep). Params: xs = 0, sep = 1.
fn join(m: &mut Mb, a: &mut Asm) {
    let (n, total, k, p, at, e, lsep) = (a.local(I32), a.local(I64), a.local(I32), a.local(I32), a.local(I32), a.local(I32), a.local(I32));
    a.all([I::LocalGet(1), I::I32Load(m32(0)), I::LocalSet(lsep)]);
    a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalTee(n), I::I32Eqz, I::If(BlockType::Empty), I::I32Const(0), I::Call(m.rt(Rt::StrNew)), I::Return, I::End]);
    // total = lsep * (n - 1) + sum of element lengths, in i64 so that huge results cannot wrap
    a.all([I::LocalGet(lsep), I::I64ExtendI32U, I::LocalGet(n), I::I32Const(1), I::I32Sub, I::I64ExtendI32U, I::I64Mul, I::LocalSet(total)]);
    loop_until(a, k, n);
    slot_addr(a, 0, k);
    a.all([I::I64Load(m64(8)), I::I32WrapI64, I::I32Load(m32(0)), I::I64ExtendI32U, I::LocalGet(total), I::I64Add, I::LocalSet(total)]);
    loop_next(a, k);
    // Beyond 2 GiB the i32 length cannot be stored; such a request always exceeds the memory limit.
    a.all([I::LocalGet(total), I::I64Const(0x7FFF_0000), I::I64GtU, I::If(BlockType::Empty), I::LocalGet(total), I::Call(m.rt(Rt::Alloc)), I::Drop, I::End]);
    a.all([I::LocalGet(total), I::I32WrapI64, I::Call(m.rt(Rt::StrNew)), I::LocalTee(p), I::I32Const(4), I::I32Add, I::LocalSet(at), I::I32Const(0), I::LocalSet(k)]);
    loop_until(a, k, n);
    a.all([I::LocalGet(k), I::If(BlockType::Empty)]);
    a.all([I::LocalGet(at), I::LocalGet(1), I::I32Const(4), I::I32Add, I::LocalGet(lsep), copy(), I::LocalGet(at), I::LocalGet(lsep), I::I32Add, I::LocalSet(at), I::End]);
    slot_addr(a, 0, k);
    a.all([I::I64Load(m64(8)), I::I32WrapI64, I::LocalSet(e)]);
    a.all([I::LocalGet(at), I::LocalGet(e), I::I32Const(4), I::I32Add, I::LocalGet(e), I::I32Load(m32(0)), copy()]);
    a.all([I::LocalGet(at), I::LocalGet(e), I::I32Load(m32(0)), I::I32Add, I::LocalSet(at)]);
    loop_next(a, k);
    a.i(I::LocalGet(p));
}

/// to_int(s): optional sign, then decimal digits only; None on anything else or on overflow.
/// Accumulates negatively so that i64::MIN parses.
fn str_to_int(m: &mut Mb, a: &mut Asm) {
    let (n, i, neg, acc, d) = (a.local(I32), a.local(I32), a.local(I32), a.local(I64), a.local(I64));
    let none = m.tag_addr(TAG_NONE);
    let ret_none = [I::I32Const(none), I::Return];
    a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalTee(n), I::I32Eqz, I::If(BlockType::Empty)]);
    a.all(ret_none.clone());
    a.i(I::End);
    let first = [I::LocalGet(0), I::I32Load8U(m8(4))];
    a.all(first.clone());
    a.all([I::I32Const(b'-' as i32), I::I32Eq, I::If(BlockType::Empty), I::I32Const(1), I::LocalSet(neg), I::I32Const(1), I::LocalSet(i), I::End]);
    a.all(first);
    a.all([I::I32Const(b'+' as i32), I::I32Eq, I::If(BlockType::Empty), I::I32Const(1), I::LocalSet(i), I::End]);
    a.all([I::LocalGet(i), I::LocalGet(n), I::I32Eq, I::If(BlockType::Empty)]);
    a.all(ret_none.clone());
    a.i(I::End);
    loop_until(a, i, n);
    a.all([I::LocalGet(0), I::LocalGet(i), I::I32Add, I::I32Load8U(m8(4)), I::I32Const(48), I::I32Sub, I::I64ExtendI32U, I::LocalTee(d), I::I64Const(9), I::I64GtU]);
    // acc < MIN / 10 means acc * 10 overflows; acc * 10 < MIN + d means acc * 10 - d overflows
    a.all([I::LocalGet(acc), I::I64Const(i64::MIN / 10), I::I64LtS, I::I32Or, I::If(BlockType::Empty)]);
    a.all(ret_none.clone());
    a.i(I::End);
    a.all([I::LocalGet(acc), I::I64Const(10), I::I64Mul, I::LocalTee(acc), I::I64Const(i64::MIN), I::LocalGet(d), I::I64Add, I::I64LtS, I::If(BlockType::Empty)]);
    a.all(ret_none.clone());
    a.i(I::End);
    a.all([I::LocalGet(acc), I::LocalGet(d), I::I64Sub, I::LocalSet(acc)]);
    loop_next(a, i);
    a.all([I::LocalGet(neg), I::I32Eqz, I::If(BlockType::Empty)]);
    a.all([I::LocalGet(acc), I::I64Const(i64::MIN), I::I64Eq, I::If(BlockType::Empty)]);
    a.all(ret_none);
    a.all([I::End, I::I64Const(0), I::LocalGet(acc), I::I64Sub, I::LocalSet(acc), I::End]);
    a.all([I::I32Const(TAG_SOME as i32), I::LocalGet(acc), I::Call(m.rt(Rt::Variant1))]);
}

/// from_int(n): decimal text. Works on the non-positive value so that i64::MIN needs no special case.
fn int_to_str(m: &mut Mb, a: &mut Asm) {
    let (neg, v, t, digits, len, p, pos) = (a.local(I32), a.local(I64), a.local(I64), a.local(I32), a.local(I32), a.local(I32), a.local(I32));
    a.all([I::LocalGet(0), I::I64Const(0), I::I64LtS, I::LocalTee(neg)]);
    a.all([I::If(BlockType::Result(I64)), I::LocalGet(0), I::Else, I::I64Const(0), I::LocalGet(0), I::I64Sub, I::End, I::LocalTee(v), I::LocalSet(t)]);
    a.all([I::I32Const(1), I::LocalSet(digits)]);
    a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty)]);
    a.all([I::LocalGet(t), I::I64Const(10), I::I64DivS, I::LocalTee(t), I::I64Eqz, I::BrIf(1)]);
    incr(a, digits);
    a.all([I::Br(0), I::End, I::End]);
    a.all([I::LocalGet(digits), I::LocalGet(neg), I::I32Add, I::LocalTee(len), I::Call(m.rt(Rt::StrNew)), I::LocalSet(p)]);
    a.all([I::LocalGet(len), I::I32Const(1), I::I32Sub, I::LocalSet(pos), I::LocalGet(v), I::LocalSet(t)]);
    a.all([I::Loop(BlockType::Empty)]);
    a.all([I::LocalGet(p), I::LocalGet(pos), I::I32Add, I::I32Const(48), I::I64Const(0), I::LocalGet(t), I::I64Const(10), I::I64RemS, I::I64Sub, I::I32WrapI64, I::I32Add, I::I32Store8(m8(4))]);
    a.all([I::LocalGet(pos), I::I32Const(1), I::I32Sub, I::LocalSet(pos)]);
    a.all([I::LocalGet(t), I::I64Const(10), I::I64DivS, I::LocalTee(t), I::I64Const(0), I::I64Ne, I::BrIf(0), I::End]);
    a.all([I::LocalGet(neg), I::If(BlockType::Empty), I::LocalGet(p), I::I32Const(b'-' as i32), I::I32Store8(m8(4)), I::End, I::LocalGet(p)]);
}

/// sort_int(xs): heapsort on a copy (O(n log n), no extra memory).
fn sort_int(m: &mut Mb, a: &mut Asm) {
    let (n, p, base, i) = (a.local(I32), a.local(I32), a.local(I32), a.local(I32));
    let sift = m.rt(Rt::Sift);
    a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalTee(n), I::Call(m.rt(Rt::ListNew)), I::LocalTee(p), I::I32Const(8), I::I32Add, I::LocalTee(base)]);
    a.all([I::LocalGet(0), I::I32Const(8), I::I32Add, I::LocalGet(n), I::I32Const(3), I::I32Shl, copy()]);
    // heapify: i = n/2 - 1 down to 0
    a.all([I::LocalGet(n), I::I32Const(1), I::I32ShrU, I::I32Const(1), I::I32Sub, I::LocalSet(i)]);
    a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty), I::LocalGet(i), I::I32Const(0), I::I32LtS, I::BrIf(1)]);
    a.all([I::LocalGet(base), I::LocalGet(i), I::LocalGet(n), I::Call(sift), I::LocalGet(i), I::I32Const(1), I::I32Sub, I::LocalSet(i), I::Br(0), I::End, I::End]);
    // pop: i = n - 1 down to 1: swap(0, i); sift(0, i)
    let t = a.local(I64);
    a.all([I::LocalGet(n), I::I32Const(1), I::I32Sub, I::LocalSet(i)]);
    a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty), I::LocalGet(i), I::I32Const(0), I::I32LeS, I::BrIf(1)]);
    a.all([I::LocalGet(base), I::I64Load(m64(0)), I::LocalSet(t)]);
    a.all([I::LocalGet(base), I::LocalGet(base), I::LocalGet(i), I::I32Const(3), I::I32Shl, I::I32Add, I::I64Load(m64(0)), I::I64Store(m64(0))]);
    a.all([I::LocalGet(base), I::LocalGet(i), I::I32Const(3), I::I32Shl, I::I32Add, I::LocalGet(t), I::I64Store(m64(0))]);
    a.all([I::LocalGet(base), I::I32Const(0), I::LocalGet(i), I::Call(sift), I::LocalGet(i), I::I32Const(1), I::I32Sub, I::LocalSet(i), I::Br(0), I::End, I::End]);
    a.i(I::LocalGet(p));
}

/// Sift-down for a max-heap of i64 at `base` (params: base = 0, root = 1, end = 2).
fn sift(a: &mut Asm) {
    let (child, t) = (a.local(I32), a.local(I64));
    let at = |a: &mut Asm, idx: u32| a.all([I::LocalGet(0), I::LocalGet(idx), I::I32Const(3), I::I32Shl, I::I32Add]);
    a.i(I::Loop(BlockType::Empty));
    a.all([I::LocalGet(1), I::I32Const(1), I::I32Shl, I::I32Const(1), I::I32Add, I::LocalTee(child), I::LocalGet(2), I::I32GeS, I::If(BlockType::Empty), I::Return, I::End]);
    // pick the larger child
    a.all([I::LocalGet(child), I::I32Const(1), I::I32Add, I::LocalGet(2), I::I32LtS, I::If(BlockType::Empty)]);
    at(a, child);
    a.i(I::I64Load(m64(0)));
    at(a, child);
    a.all([I::I64Load(m64(8)), I::I64LtS, I::If(BlockType::Empty), I::LocalGet(child), I::I32Const(1), I::I32Add, I::LocalSet(child), I::End, I::End]);
    at(a, 1);
    a.i(I::I64Load(m64(0)));
    at(a, child);
    a.all([I::I64Load(m64(0)), I::I64GeS, I::If(BlockType::Empty), I::Return, I::End]);
    // swap root and child, continue at child
    at(a, 1);
    a.all([I::I64Load(m64(0)), I::LocalSet(t)]);
    at(a, 1);
    at(a, child);
    a.all([I::I64Load(m64(0)), I::I64Store(m64(0))]);
    at(a, child);
    a.all([I::LocalGet(t), I::I64Store(m64(0)), I::LocalGet(child), I::LocalSet(1), I::Br(0), I::End]);
}

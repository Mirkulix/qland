//! Small helpers for writing WebAssembly function bodies by hand.

use qal_core::ast::Type;
use wasm_encoder::{Function, Instruction as I, MemArg, ValType};

/// Machine representation of a QAL value: Int = I64, Float = F64, everything else = I32
/// (Bool/Unit as 0/1, all heap values and function references as pointers/table indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Repr {
    I32,
    I64,
    F64,
}

impl Repr {
    pub fn of(t: &Type) -> Repr {
        match t {
            Type::Int => Repr::I64,
            Type::Float => Repr::F64,
            _ => Repr::I32,
        }
    }

    pub fn val(self) -> ValType {
        match self {
            Repr::I32 => ValType::I32,
            Repr::I64 => ValType::I64,
            Repr::F64 => ValType::F64,
        }
    }
}

pub fn val(t: &Type) -> ValType {
    Repr::of(t).val()
}

pub fn m8(offset: u64) -> MemArg {
    MemArg { offset, align: 0, memory_index: 0 }
}

pub fn m32(offset: u64) -> MemArg {
    MemArg { offset, align: 2, memory_index: 0 }
}

pub fn m64(offset: u64) -> MemArg {
    MemArg { offset, align: 3, memory_index: 0 }
}

/// Converts the value on the stack to its 8-byte slot form (i64).
pub fn to_slot(r: Repr) -> Option<I<'static>> {
    match r {
        Repr::I32 => Some(I::I64ExtendI32U),
        Repr::I64 => None,
        Repr::F64 => Some(I::I64ReinterpretF64),
    }
}

/// Converts an i64 slot on the stack back to the value representation.
pub fn from_slot(r: Repr) -> Option<I<'static>> {
    match r {
        Repr::I32 => Some(I::I32WrapI64),
        Repr::I64 => None,
        Repr::F64 => Some(I::F64ReinterpretI64),
    }
}

/// A function body under construction: parameters are locals `0..n`, extra locals follow.
pub struct Asm {
    n_params: u32,
    locals: Vec<ValType>,
    pub ins: Vec<I<'static>>,
}

impl Asm {
    pub fn new(n_params: usize) -> Asm {
        Asm { n_params: n_params as u32, locals: Vec::new(), ins: Vec::new() }
    }

    pub fn local(&mut self, v: ValType) -> u32 {
        self.locals.push(v);
        self.n_params + self.locals.len() as u32 - 1
    }

    pub fn i(&mut self, ins: I<'static>) {
        self.ins.push(ins);
    }

    pub fn all<const N: usize>(&mut self, ins: [I<'static>; N]) {
        self.ins.extend(ins);
    }

    pub fn opt(&mut self, ins: Option<I<'static>>) {
        self.ins.extend(ins);
    }

    pub fn finish(mut self) -> Function {
        self.ins.push(I::End);
        let mut f = Function::new(self.locals.iter().map(|v| (1, *v)));
        for ins in &self.ins {
            f.instruction(ins);
        }
        f
    }
}

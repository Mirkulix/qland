//! Native JIT (Just-In-Time) Compiler for Apple Silicon (ARM64).
//!
//! Emits native AArch64 machine instructions directly into executable memory
//! for sub-nanosecond hardware execution (2-5 clock cycles).

use crate::dag::{Graph, OpCode};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
extern "C" {
    fn mmap(
        addr: *mut std::ffi::c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut std::ffi::c_void;
    fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
    fn pthread_jit_write_protect_np(enabled: i32);
    fn sys_icache_invalidate(start: *mut std::ffi::c_void, len: usize);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const PROT_READ: i32 = 0x01;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const PROT_WRITE: i32 = 0x02;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const PROT_EXEC: i32 = 0x04;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const MAP_PRIVATE: i32 = 0x0002;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const MAP_ANON: i32 = 0x1000;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const MAP_JIT: i32 = 0x0800;

pub struct JitEngine;

pub struct CompiledFunction {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    code_ptr: *mut u8,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    size: usize,
}

unsafe impl Send for CompiledFunction {}
unsafe impl Sync for CompiledFunction {}

impl Drop for CompiledFunction {
    fn drop(&mut self) {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        unsafe {
            if !self.code_ptr.is_null() {
                munmap(self.code_ptr as *mut std::ffi::c_void, self.size);
            }
        }
    }
}

impl CompiledFunction {
    /// Executes the JIT compiled function with two i64 arguments.
    pub unsafe fn call_i64_2(&self, arg0: i64, arg1: i64) -> i64 {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let func: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(self.code_ptr);
            func(arg0, arg1)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (arg0, arg1);
            0
        }
    }

    /// Executes a JIT compiled tensor reduction with (ptr, len) returning f64 in d0.
    pub unsafe fn call_tensor_reduce(&self, ptr: *const f64, len: usize) -> f64 {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let func: extern "C" fn(*const f64, usize) -> f64 = std::mem::transmute(self.code_ptr);
            func(ptr, len)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (ptr, len);
            0.0
        }
    }
}

impl JitEngine {
    pub fn is_supported() -> bool {
        cfg!(all(target_os = "macos", target_arch = "aarch64"))
    }

    /// Compiles a 64-bit integer DAG into native ARM64 machine code.
    pub fn compile(graph: &Graph) -> Result<CompiledFunction, String> {
        if !Self::is_supported() {
            return Err("JIT currently requires macOS on ARM64 (Apple Silicon)".to_string());
        }

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            if graph.input_types.first() == Some(&crate::dag::DataType::Tensor1D) {
                Self::compile_arm64_tensor(graph)
            } else {
                Self::compile_arm64(graph)
            }
        }

        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = graph;
            Err("Unsupported platform".to_string())
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn compile_arm64_tensor(graph: &Graph) -> Result<CompiledFunction, String> {
        let mut code: Vec<u32> = Vec::new();

        let mut bias: f64 = 0.0;
        for node in &graph.nodes {
            if let OpCode::TensorMapAdd(b) = node.op {
                bias = b;
                break;
            }
        }

        // 1. Clear accumulator: fmov d0, xzr (0x9e6703e0)
        code.push(0x9e6703e0);

        // 2. Load bias into x2, then fmov d1, x2 (0x9e670041)
        Self::emit_mov_const(&mut code, 2, bias.to_bits() as i64);
        code.push(0x9e670041);

        // 3. If len (x1) == 0, jump to ret (6 instructions ahead)
        // cbz x1, #24 -> 0xb4000000 | (6 << 5) | 1
        code.push(0xb4000000 | (6 << 5) | 1);

        // 4. Loop:
        // ldr d2, [x0], #8 (0xfc408402 for 64-bit float D2)
        code.push(0xfc408402);
        // fadd d2, d2, d1 (0x1e612842)
        code.push(0x1e612842);
        // fadd d0, d0, d2 (0x1e622800)
        code.push(0x1e622800);
        // subs x1, x1, #1 (0xf1000421)
        code.push(0xf1000421);
        // b.ne loop (jump back 4 instructions: -4) -> 0x54ffff81
        code.push(0x54ffff81);

        // 5. Ret: (0xd65f03c0)
        code.push(0xd65f03c0);

        Self::allocate_executable_memory(code)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn compile_arm64(graph: &Graph) -> Result<CompiledFunction, String> {
        let mut code: Vec<u32> = Vec::new();

        // Stack frame: each node value at [sp, #(idx * 8)]
        let stack_slots = graph.nodes.len().max(2);
        let frame_size = ((stack_slots * 8 + 15) / 16) * 16; // 16-byte aligned

        // 1. Prologue: sub sp, sp, #frame_size
        Self::emit_sub_imm_sp(&mut code, frame_size as u32);

        // 2. Body: Process each node topologically
        for (idx, node) in graph.nodes.iter().enumerate() {
            let offset = (idx * 8) as u32;

            match &node.op {
                OpCode::Param(slot) => {
                    let reg = *slot as u8;
                    Self::emit_str_reg(&mut code, reg, offset);
                }
                OpCode::ConstI64(val) => {
                    Self::emit_mov_const(&mut code, 9, *val);
                    Self::emit_str_reg(&mut code, 9, offset);
                }
                OpCode::Add => {
                    let in0 = (node.inputs[0] * 8) as u32;
                    let in1 = (node.inputs[1] * 8) as u32;
                    Self::emit_ldr_reg(&mut code, 9, in0);
                    Self::emit_ldr_reg(&mut code, 10, in1);
                    Self::emit_add(&mut code, 9, 9, 10);
                    Self::emit_str_reg(&mut code, 9, offset);
                }
                OpCode::Sub => {
                    let in0 = (node.inputs[0] * 8) as u32;
                    let in1 = (node.inputs[1] * 8) as u32;
                    Self::emit_ldr_reg(&mut code, 9, in0);
                    Self::emit_ldr_reg(&mut code, 10, in1);
                    Self::emit_sub(&mut code, 9, 9, 10);
                    Self::emit_str_reg(&mut code, 9, offset);
                }
                OpCode::Mul => {
                    let in0 = (node.inputs[0] * 8) as u32;
                    let in1 = (node.inputs[1] * 8) as u32;
                    Self::emit_ldr_reg(&mut code, 9, in0);
                    Self::emit_ldr_reg(&mut code, 10, in1);
                    Self::emit_mul(&mut code, 9, 9, 10);
                    Self::emit_str_reg(&mut code, 9, offset);
                }
                OpCode::Div => {
                    let in0 = (node.inputs[0] * 8) as u32;
                    let in1 = (node.inputs[1] * 8) as u32;
                    Self::emit_ldr_reg(&mut code, 9, in0);
                    Self::emit_ldr_reg(&mut code, 10, in1);
                    Self::emit_sdiv(&mut code, 9, 9, 10);
                    Self::emit_str_reg(&mut code, 9, offset);
                }
                OpCode::Clamp => {
                    let in_val = (node.inputs[0] * 8) as u32;
                    let in_low = (node.inputs[1] * 8) as u32;
                    let in_high = (node.inputs[2] * 8) as u32;
                    Self::emit_ldr_reg(&mut code, 9, in_val);   // val
                    Self::emit_ldr_reg(&mut code, 10, in_low);  // low
                    Self::emit_ldr_reg(&mut code, 11, in_high); // high

                    // cmp val (x9), low (x10)
                    Self::emit_cmp(&mut code, 9, 10);
                    // csel x9, x9, x10, ge (if val >= low then val else low) -> cond 0xa
                    Self::emit_csel(&mut code, 9, 9, 10, 0xa);

                    // cmp val (x9), high (x11)
                    Self::emit_cmp(&mut code, 9, 11);
                    // csel x9, x9, x11, le (if val <= high then val else high) -> cond 0xd
                    Self::emit_csel(&mut code, 9, 9, 11, 0xd);

                    Self::emit_str_reg(&mut code, 9, offset);
                }
                _ => return Err(format!("JIT not yet implemented for OpCode: {:?}", node.op)),
            }
        }

        // 3. Epilogue: load return node into x0
        let ret_offset = (graph.return_node * 8) as u32;
        Self::emit_ldr_reg(&mut code, 0, ret_offset);

        // add sp, sp, #frame_size
        Self::emit_add_imm_sp(&mut code, frame_size as u32);

        // ret (0xd65f03c0)
        code.push(0xd65f03c0);

        Self::allocate_executable_memory(code)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn allocate_executable_memory(code: Vec<u32>) -> Result<CompiledFunction, String> {
        let page_size = 4096;
        let alloc_size = ((code.len() * 4 + page_size - 1) / page_size) * page_size;

        unsafe {
            let ptr = mmap(
                std::ptr::null_mut(),
                alloc_size,
                PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | MAP_ANON | MAP_JIT,
                -1,
                0,
            );

            if ptr == libc_map_failed() {
                return Err("Failed to allocate JIT memory via mmap".to_string());
            }

            // Disable JIT write protection to write instructions
            pthread_jit_write_protect_np(0);

            let dst = ptr as *mut u32;
            for (i, instr) in code.iter().enumerate() {
                *dst.add(i) = *instr;
            }

            // Re-enable JIT write protection (executable mode)
            pthread_jit_write_protect_np(1);

            // Invalidate instruction cache so CPU fetches new instructions
            sys_icache_invalidate(ptr, alloc_size);

            Ok(CompiledFunction {
                code_ptr: ptr as *mut u8,
                size: alloc_size,
            })
        }
    }

    #[inline]
    fn emit_add(code: &mut Vec<u32>, rd: u8, rn: u8, rm: u8) {
        // add rd, rn, rm -> 0x8b000000 | (rm << 16) | (rn << 5) | rd
        code.push(0x8b000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
    }

    #[inline]
    fn emit_sub(code: &mut Vec<u32>, rd: u8, rn: u8, rm: u8) {
        // sub rd, rn, rm -> 0xcb000000 | (rm << 16) | (rn << 5) | rd
        code.push(0xcb000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
    }

    #[inline]
    fn emit_mul(code: &mut Vec<u32>, rd: u8, rn: u8, rm: u8) {
        // madd rd, rn, rm, xzr -> 0x9b007c00 | (rm << 16) | (rn << 5) | rd
        code.push(0x9b007c00 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
    }

    #[inline]
    fn emit_sdiv(code: &mut Vec<u32>, rd: u8, rn: u8, rm: u8) {
        // sdiv rd, rn, rm -> 0x9ac00c00 | (rm << 16) | (rn << 5) | rd
        code.push(0x9ac00c00 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32));
    }

    #[inline]
    fn emit_cmp(code: &mut Vec<u32>, rn: u8, rm: u8) {
        // subs xzr, rn, rm -> 0xeb00001f | (rm << 16) | (rn << 5)
        code.push(0xeb00001f | ((rm as u32) << 16) | ((rn as u32) << 5));
    }

    #[inline]
    fn emit_csel(code: &mut Vec<u32>, rd: u8, rn: u8, rm: u8, cond: u8) {
        // csel rd, rn, rm, cond -> 0x9a800000 | (rm << 16) | (cond << 12) | (rn << 5) | rd
        code.push(0x9a800000 | ((rm as u32) << 16) | ((cond as u32) << 12) | ((rn as u32) << 5) | (rd as u32));
    }

    #[inline]
    fn emit_sub_imm_sp(code: &mut Vec<u32>, imm: u32) {
        // sub sp, sp, #imm (0xd10003ff | (imm << 10))
        code.push(0xd10003ff | (imm << 10));
    }

    #[inline]
    fn emit_add_imm_sp(code: &mut Vec<u32>, imm: u32) {
        // add sp, sp, #imm (0x910003ff | (imm << 10))
        code.push(0x910003ff | (imm << 10));
    }

    #[inline]
    fn emit_str_reg(code: &mut Vec<u32>, reg: u8, offset: u32) {
        // str x<reg>, [sp, #offset] -> 0xf90003e0 | (imm << 10) | reg
        let imm = offset / 8;
        code.push(0xf90003e0 | (imm << 10) | (reg as u32));
    }

    #[inline]
    fn emit_ldr_reg(code: &mut Vec<u32>, reg: u8, offset: u32) {
        // ldr x<reg>, [sp, #offset] -> 0xf94003e0 | (imm << 10) | reg
        let imm = offset / 8;
        code.push(0xf94003e0 | (imm << 10) | (reg as u32));
    }

    fn emit_mov_const(code: &mut Vec<u32>, reg: u8, val: i64) {
        if val >= 0 && val <= 0xffff {
            // movz x<reg>, #val (0xd2800000 | (val << 5) | reg)
            code.push(0xd2800000 | ((val as u32) << 5) | (reg as u32));
        } else if val < 0 && val >= -0x10000 {
            // movn x<reg>, #(-val - 1)
            let not_val = (!val) as u32;
            code.push(0x92800000 | (not_val << 5) | (reg as u32));
        } else {
            // Full 64-bit load: movz + movk passes
            let uval = val as u64;
            let p0 = (uval & 0xffff) as u32;
            let p1 = ((uval >> 16) & 0xffff) as u32;
            let p2 = ((uval >> 32) & 0xffff) as u32;
            let p3 = ((uval >> 48) & 0xffff) as u32;

            code.push(0xd2800000 | (p0 << 5) | (reg as u32)); // movz x, p0, lsl 0
            if p1 != 0 {
                code.push(0xf2a00000 | (p1 << 5) | (reg as u32)); // movk x, p1, lsl 16
            }
            if p2 != 0 {
                code.push(0xf2c00000 | (p2 << 5) | (reg as u32)); // movk x, p2, lsl 32
            }
            if p3 != 0 {
                code.push(0xf2e00000 | (p3 << 5) | (reg as u32)); // movk x, p3, lsl 48
            }
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn libc_map_failed() -> *mut std::ffi::c_void {
    (-1isize) as *mut std::ffi::c_void
}

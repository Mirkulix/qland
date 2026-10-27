.set MB_MAGIC, 0x1BADB002
.set MB_FLAGS, 0
.set MB_CHECK, -(MB_MAGIC + MB_FLAGS)

.section .multiboot, "a"
.align 4
.long MB_MAGIC
.long MB_FLAGS
.long MB_CHECK

.section .text
.global _start
.code32
_start:
    cli
    mov $stack_top, %esp
    mov $msg, %esi

    # COM1: 115200 8N1
    mov $0x3F8 + 1, %dx
    xor %al, %al
    out %al, %dx
    mov $0x3F8 + 3, %dx
    mov $0x80, %al
    out %al, %dx
    mov $0x3F8, %dx
    mov $0x03, %al
    out %al, %dx
    mov $0x3F8 + 1, %dx
    xor %al, %al
    out %al, %dx
    mov $0x3F8 + 3, %dx
    mov $0x03, %al
    out %al, %dx
    mov $0x3F8 + 2, %dx
    mov $0xC7, %al
    out %al, %dx
    mov $0x3F8 + 4, %dx
    mov $0x0B, %al
    out %al, %dx

print:
    lodsb
    test %al, %al
    jz hang
    mov %al, %bl
    mov $0x3F8 + 5, %dx
wait_tx:
    in %dx, %al
    test $0x20, %al
    jz wait_tx
    mov $0x3F8, %dx
    mov %bl, %al
    out %al, %dx
    jmp print

hang:
    hlt
    jmp hang

.section .rodata
msg:
    .asciz "QAL-OS M0: bare-metal kernel booted\nNo Linux. No POSIX. Agent substrate alive.\nNext: memory, interrupts, scheduler, capabilities.\n"

.section .bss
.align 16
stack:
    .skip 4096
stack_top:

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

    # GDT: null, kcode, kdata, ucode, udata, tss
    lgdt gdt_ptr
    mov $0x10, %ax
    mov %ax, %ds
    mov %ax, %es
    mov %ax, %fs
    mov %ax, %gs
    mov %ax, %ss

    # TSS descriptor
    movw $103, gdt + 40
    movl $tss, %eax
    movw %ax, gdt + 42
    shr $16, %eax
    movb %al, gdt + 44
    shr $8, %eax
    movb %al, gdt + 47
    movb $0x89, gdt + 45
    movb $0x00, gdt + 46

    ljmp $0x08, $gdt_flush
gdt_flush:
    mov $0x28, %ax
    ltr %ax

    # IDT: timer 32, syscall 0x80
    mov $idt, %edi
    xor %eax, %eax
    mov $2048, %ecx
    rep stosb
    mov $irq0, %eax
    mov %ax, idt + 32*8
    shr $16, %eax
    mov %ax, idt + 32*8 + 6
    movw $0x08, idt + 32*8 + 2
    movb $0x8E, idt + 32*8 + 5
    mov $syscall, %eax
    mov %ax, idt + 0x80*8
    shr $16, %eax
    mov %ax, idt + 0x80*8 + 6
    movw $0x08, idt + 0x80*8 + 2
    movb $0xEE, idt + 0x80*8 + 5
    lidt idt_ptr

    # PIC + PIT (timer stays masked off for the user-mode demo)
    mov $0x11, %al
    out %al, $0x20
    out %al, $0xA0
    mov $0x20, %al
    out %al, $0x21
    mov $0x28, %al
    out %al, $0xA1
    mov $0x04, %al
    out %al, $0x21
    mov $0x02, %al
    out %al, $0xA1
    mov $0x01, %al
    out %al, $0x21
    out %al, $0xA1
    mov $0xFE, %al
    out %al, $0x21
    mov $0xFF, %al
    out %al, $0xA1
    mov $0x36, %al
    out %al, $0x43
    mov $0x9B, %al
    out %al, $0x40
    mov $0x2E, %al
    out %al, $0x40
    mov $0xFF, %al
    out %al, $0x21

    # COM1
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

    # two pages, user-accessible, identity-mapped first 4 MiB
    mov $page_table, %edi
    mov $0x3, %eax
    mov $512, %ecx
.fill_low:
    mov %eax, (%edi)
    add $0x1000, %eax
    add $4, %edi
    loop .fill_low
    mov $0x200000 | 0x7, %eax
    mov $512, %ecx
.fill_high:
    mov %eax, (%edi)
    add $0x1000, %eax
    add $4, %edi
    loop .fill_high
    mov $page_dir, %edi
    mov $page_table, %eax
    or $0x7, %eax
    mov %eax, (%edi)
    mov $page_dir, %eax
    mov %eax, %cr3
    mov %cr0, %eax
    or $0x80000000, %eax
    mov %eax, %cr0

    # copy user blob to 0x200000
    mov $user_blob, %esi
    mov $0x200000, %edi
    mov $256, %ecx
    rep movsb
    mov $qal_prog, %esi
    mov $0x200100, %edi
    mov $10, %ecx
    rep movsb

    mov $boot_msg, %esi
    call serial_print

    # enter ring3 at 0x200000 with stack 0x300000
    push $0x23
    push $0x300000
    push $0x202
    push $0x1B
    push $0x200000
    iret

# timer ISR: EOI only (scheduler is off while the ring3 agent runs)
irq0:
    pusha
    mov $0x20, %dx
    mov $0x20, %al
    out %al, %dx
    popa
    iret

# syscall 2: run a tiny QAL-0 bytecode program at ESI
syscall:
    pusha
    cmp $2, %eax
    je .sys_qal
    cmp $3, %eax
    je .sys_send
    cmp $4, %eax
    je .sys_revoke
    cmp $5, %eax
    je .sys_recv
    jmp .sys_done
.sys_qal:
    cmp $0x200000, %ebx
    jb .sys_bad
    cmp $0x400000, %ebx
    jae .sys_bad
    lea 64(%ebx), %eax
    cmp $0x400000, %eax
    ja .sys_bad
    mov %ebx, %esi
    mov $qbuf, %edi
    mov $64, %ecx
    rep movsb
    mov $qbuf, %esi
    call qal0_run
    jmp .sys_done
.sys_bad:
    mov $bad_ptr, %esi
    call serial_print
    jmp .sys_done
.sys_send:
    cmp $1, %ebx
    je .send1
    cmp $2, %ebx
    je .send2
    mov $cap_denied, %esi
    call serial_print
    jmp .sys_done
.send1:
    cmpl $0, ep1_live
    je .denied
    movl $0x51, ep1_mail
    mov $cap1_ok, %esi
    call serial_print
    jmp .sys_done
.send2:
    cmpl $0, ep2_live
    je .denied
    movl $0x52, ep2_mail
    mov $cap2_ok, %esi
    call serial_print
    jmp .sys_done
.denied:
    mov $cap_denied, %esi
    call serial_print
    jmp .sys_done
.sys_revoke:
    cmp $1, %ebx
    jne .revoke2
    movl $0, ep1_live
    mov $cap1_revoked, %esi
    call serial_print
    jmp .sys_done
.revoke2:
    cmp $2, %ebx
    jne .sys_done
    movl $0, ep2_live
    mov $cap2_revoked, %esi
    call serial_print
    jmp .sys_done
.sys_recv:
    cmp $1, %ebx
    je .recv1
    cmp $2, %ebx
    je .recv2
    jmp .recv_empty
.recv1:
    cmpl $0, ep1_live
    je .recv_empty
    mov ep1_mail, %eax
    test %eax, %eax
    jz .recv_empty
    cmp $0x200000, %ecx
    jb .recv_bad
    cmp $0x400000, %ecx
    jae .recv_bad
    mov %eax, (%ecx)
    movl $0, ep1_mail
    mov $recv1_msg, %esi
    call serial_print
    jmp .sys_done
.recv2:
    cmpl $0, ep2_live
    je .recv_empty
    mov ep2_mail, %eax
    test %eax, %eax
    jz .recv_empty
    cmp $0x200000, %ecx
    jb .recv_bad
    cmp $0x400000, %ecx
    jae .recv_bad
    mov %eax, (%ecx)
    movl $0, ep2_mail
    mov $recv2_msg, %esi
    call serial_print
    jmp .sys_done
.recv_empty:
    mov $recv_empty_msg, %esi
    call serial_print
    jmp .sys_done
.recv_bad:
    mov $recv_bad_msg, %esi
    call serial_print
.sys_done:
    popa
    iret

# QAL-1: push imm8, add 0x10, mul 0x11, sub 0x12, print 0x20, halt 0xFF
qal0_run:
    xor %ecx, %ecx
    mov $64, %edx
.q_loop:
    test %edx, %edx
    jz .q_halt
    dec %edx
    lodsb
    cmp $0xFF, %al
    je .q_halt
    cmp $0x01, %al
    je .q_push
    cmp $0x10, %al
    je .q_add
    cmp $0x11, %al
    je .q_mul
    cmp $0x12, %al
    je .q_sub
    cmp $0x20, %al
    je .q_print
    jmp .q_loop
.q_push:
    lodsb
    mov %al, qstack(%ecx)
    inc %ecx
    jmp .q_loop
.q_add:
    dec %ecx
    mov qstack(%ecx), %al
    dec %ecx
    add qstack(%ecx), %al
    mov %al, qstack(%ecx)
    inc %ecx
    jmp .q_loop
.q_mul:
    dec %ecx
    mov qstack(%ecx), %al
    dec %ecx
    mulb qstack(%ecx)
    mov %al, qstack(%ecx)
    inc %ecx
    jmp .q_loop
.q_sub:
    dec %ecx
    mov qstack(%ecx), %al
    dec %ecx
    sub %al, qstack(%ecx)
    inc %ecx
    jmp .q_loop
.q_print:
    dec %ecx
    mov qstack(%ecx), %al
    add $'0', %al
    mov %al, outbuf
    push %esi
    mov $qal_msg, %esi
    call serial_print
    mov $outbuf, %esi
    call serial_print
    mov $newline, %esi
    call serial_print
    pop %esi
    jmp .q_loop
.q_halt:
    ret

serial_print:
    lodsb
    test %al, %al
    jz .done
    mov %al, %bl
    mov $0x3F8 + 5, %dx
.wait:
    in %dx, %al
    test $0x20, %al
    jz .wait
    mov $0x3F8, %dx
    mov %bl, %al
    out %al, %dx
    jmp serial_print
.done:
    ret

.section .rodata
boot_msg:
    .asciz "QAL-OS M6: receive/copy-out, QAL-1 interpreter, endpoints\n"
qal_msg:
    .asciz "agent: qal-1 = "
newline:
    .asciz "\n"
bad_ptr:
    .asciz "[syscall: bad user ptr]\n"
cap1_ok:
    .asciz "[ep1: send ok]\n"
cap2_ok:
    .asciz "[ep2: send ok]\n"
cap_denied:
    .asciz "[ep1: denied]\n"
cap1_revoked:
    .asciz "[ep1: revoked]\n"
cap2_revoked:
    .asciz "[ep2: revoked]\n"
recv1_msg:
    .asciz "[ep1: recv 0x51]\n"
recv2_msg:
    .asciz "[ep2: recv 0x52]\n"
recv_empty_msg:
    .asciz "[recv: empty]\n"
recv_bad_msg:
    .asciz "[recv: bad user ptr]\n"
qal_prog:
    .byte 0x01, 0x01, 0x01, 0x02, 0x10, 0x01, 0x03, 0x11, 0x20, 0xFF

user_blob:
    # valid QAL-1 program: (1+2)*3 = 9
    .byte 0xBC, 0x00, 0x00, 0x30, 0x00   # mov $0x300000, %esp
    .byte 0xBB, 0x00, 0x01, 0x20, 0x00   # mov $0x200080, %ebx
    .byte 0xB8, 0x02, 0x00, 0x00, 0x00
    .byte 0xCD, 0x80
    # bad kernel pointer
    .byte 0xBB, 0x00, 0x00, 0x10, 0x00
    .byte 0xB8, 0x02, 0x00, 0x00, 0x00
    .byte 0xCD, 0x80
    # cap1 send / revoke / denied
    .byte 0xBB, 0x01, 0x00, 0x00, 0x00
    .byte 0xB8, 0x03, 0x00, 0x00, 0x00
    .byte 0xCD, 0x80
    .byte 0xBB, 0x01, 0x00, 0x00, 0x00
    .byte 0xB8, 0x04, 0x00, 0x00, 0x00
    .byte 0xCD, 0x80
    .byte 0xBB, 0x01, 0x00, 0x00, 0x00
    .byte 0xB8, 0x03, 0x00, 0x00, 0x00
    .byte 0xCD, 0x80
    # cap2 send then receive (copy-out to 0x300100)
    .byte 0xBB, 0x02, 0x00, 0x00, 0x00
    .byte 0xB8, 0x03, 0x00, 0x00, 0x00
    .byte 0xCD, 0x80
    .byte 0xBB, 0x02, 0x00, 0x00, 0x00
    .byte 0xB9, 0x00, 0x01, 0x30, 0x00
    .byte 0xB8, 0x05, 0x00, 0x00, 0x00
    .byte 0xCD, 0x80
    .byte 0xEB, 0xFE
        .skip 32

.section .data
.align 4
ep1_live: .long 1
ep2_live: .long 1
ep1_mail: .long 0
ep2_mail: .long 0
.align 8
gdt:
    .quad 0x0000000000000000
    .quad 0x00CF9A000000FFFF
    .quad 0x00CF92000000FFFF
    .quad 0x00CFFA000000FFFF
    .quad 0x00CFF2000000FFFF
    .quad 0x0000000000000000
    .quad 0x0000000000000000
gdt_end:
gdt_ptr:
    .word gdt_end - gdt - 1
    .long gdt

.align 8
idt:
    .skip 2048
idt_end:
idt_ptr:
    .word idt_end - idt - 1
    .long idt

.align 4
tss:
    .long 0
    .long stack_top
    .long 0x10
    .skip 88
    .word 0
    .word 104

.section .bss
.align 4096
page_dir:
    .skip 4096
page_table:
    .skip 4096
.align 16
stack:
    .skip 4096
stack_top:
.align 16
qbuf:
    .skip 64
qstack:
    .skip 16
outbuf:
    .skip 2

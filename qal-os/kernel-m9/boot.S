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

    lgdt gdt_ptr
    mov $0x10, %ax
    mov %ax, %ds
    mov %ax, %es
    mov %ax, %fs
    mov %ax, %gs
    mov %ax, %ss

    # TSS descriptor (index 5)
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

    # PIC remap, only IRQ0
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

    # PIT 100 Hz
    mov $0x36, %al
    out %al, $0x43
    mov $0x9B, %al
    out %al, $0x40
    mov $0x2E, %al
    out %al, $0x40

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

    # paging: 0-2 MiB supervisor, 2-4 MiB user
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

    # user code A at 0x200000, B at 0x201000
    mov $user_a, %esi
    mov $0x200000, %edi
    mov $64, %ecx
    rep movsb
    mov $user_b, %esi
    mov $0x201000, %edi
    mov $64, %ecx
    rep movsb
    mov $qal_prog_a, %esi
    mov $0x200100, %edi
    mov $10, %ecx
    rep movsb
    mov $qal_prog_b, %esi
    mov $0x201100, %edi
    mov $10, %ecx
    rep movsb

    # build kernel frames for A and B
    # frame: edi esi ebp esp ebx edx ecx eax eip cs eflags uesp uss
    mov $kernel_stack_a_top, %eax
    sub $52, %eax
    movl $0, 0(%eax)
    movl $0, 4(%eax)
    movl $0, 8(%eax)
    movl $0, 12(%eax)
    movl $0, 16(%eax)
    movl $0, 20(%eax)
    movl $0, 24(%eax)
    movl $0, 28(%eax)
    movl $0x200000, 32(%eax)
    movl $0x1B, 36(%eax)
    movl $0x202, 40(%eax)
    movl $0x300000, 44(%eax)
    movl $0x23, 48(%eax)
    mov %eax, task_a_esp

    mov $kernel_stack_b_top, %eax
    sub $52, %eax
    movl $0, 0(%eax)
    movl $0, 4(%eax)
    movl $0, 8(%eax)
    movl $0, 12(%eax)
    movl $0, 16(%eax)
    movl $0, 20(%eax)
    movl $0, 24(%eax)
    movl $0, 28(%eax)
    movl $0x201000, 32(%eax)
    movl $0x1B, 36(%eax)
    movl $0x202, 40(%eax)
    movl $0x301000, 44(%eax)
    movl $0x23, 48(%eax)
    mov %eax, task_b_esp
    mov %eax, next_esp

    # TSS esp0 for A
    movl $kernel_stack_a_top, tss + 4

    mov $boot_msg, %esi
    call serial_print

    # enter ring3 task A
    mov task_a_esp, %esp
    popa
    iret

# timer ISR: ring3 -> ring0, switch kernel frame and TSS esp0
irq0:
    pusha
    mov %esp, current_esp
    mov next_esp, %esp
    cmp task_a_esp, %esp
    je .to_a
    mov task_a_esp, %eax
    mov %eax, next_esp
    movl $kernel_stack_b_top, tss + 4
    jmp .eoi
.to_a:
    mov task_b_esp, %eax
    mov %eax, next_esp
    movl $kernel_stack_a_top, tss + 4
.eoi:
    mov $0x20, %dx
    mov $0x20, %al
    out %al, %dx
    popa
    iret

# syscall 6: putc(ebx)
syscall:
    pusha
    cli
    cmp $2, %eax
    je .sys_qal
    cmp $6, %eax
    je .sys_putc
    cmp $7, %eax
    je .sys_yield
    jmp .sys_done
.sys_putc:
    mov $0x3F8 + 5, %dx
.sys_wait:
    in %dx, %al
    test $0x20, %al
    jz .sys_wait
    mov $0x3F8, %dx
    mov %bl, %al
    out %al, %dx
    jmp .sys_done
.sys_qal:
    cmp $0x200000, %ebx
    jb .sys_done
    cmp $0x400000, %ebx
    jae .sys_done
    lea 64(%ebx), %eax
    cmp $0x400000, %eax
    ja .sys_done
    mov %ebx, %esi
    mov $qbuf, %edi
    mov $64, %ecx
    rep movsb
    mov $qbuf, %esi
    call qal0_run
    jmp .sys_done
.sys_yield:
    mov %esp, current_esp
    mov next_esp, %esp
    cmp task_a_esp, %esp
    je .y_target_a
    # target B: next is A, TSS stack A
    mov task_a_esp, %eax
    mov %eax, next_esp
    movl $kernel_stack_b_top, tss + 4
    jmp .sys_done
.y_target_a:
    # target A: next is B, TSS stack A
    mov task_b_esp, %eax
    mov %eax, next_esp
    movl $kernel_stack_a_top, tss + 4
.sys_done:
    popa
    iret


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
    jz .sp_done
    mov %al, %bl
    mov $0x3F8 + 5, %dx
.sp_wait:
    in %dx, %al
    test $0x20, %al
    jz .sp_wait
    mov $0x3F8, %dx
    mov %bl, %al
    out %al, %dx
    jmp serial_print
.sp_done:
    ret

.section .rodata
boot_msg:
    .asciz "QAL-OS M9: QAL-1 in two preemptive ring3 tasks\n"
qal_msg:
    .asciz "[qal-1 = "
newline:
    .asciz "]\n"
user_a:
    .byte 0xBC, 0x00, 0x00, 0x30, 0x00   # esp
    .byte 0xBB, 0x00, 0x01, 0x20, 0x00   # program 0x200100
    .byte 0xB8, 0x02, 0x00, 0x00, 0x00
    .byte 0xCD, 0x80                      # qal-1
    .byte 0xBB, 0x41, 0x00, 0x00, 0x00
    .byte 0xB8, 0x06, 0x00, 0x00, 0x00
    .byte 0xCD, 0x80                      # putc A
    .byte 0xEB, 0xF2                      # loop putc
    .skip 33
user_b:
    .byte 0xBC, 0x00, 0x10, 0x30, 0x00
    .byte 0xBB, 0x00, 0x11, 0x20, 0x00   # program 0x201100
    .byte 0xB8, 0x02, 0x00, 0x00, 0x00
    .byte 0xCD, 0x80
    .byte 0xBB, 0x42, 0x00, 0x00, 0x00
    .byte 0xB8, 0x06, 0x00, 0x00, 0x00
    .byte 0xCD, 0x80
    .byte 0xEB, 0xF2
    .skip 33
qal_prog_a:
    .byte 0x01, 0x01, 0x01, 0x02, 0x10, 0x01, 0x03, 0x11, 0x20, 0xFF
qal_prog_b:
    .byte 0x01, 0x02, 0x01, 0x03, 0x10, 0x20, 0xFF, 0x00, 0x00, 0x00

.section .data
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
    .long 0
    .long 0x10
    .skip 88
    .word 0
    .word 104

.align 4
current_esp: .long 0
next_esp: .long 0
task_a_esp: .long 0
task_b_esp: .long 0

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
kernel_stack_a:
    .skip 4096
kernel_stack_a_top:
.align 16
kernel_stack_b:
    .skip 4096
kernel_stack_b_top:
.align 16
qbuf:
    .skip 64
qstack:
    .skip 16
outbuf:
    .skip 2

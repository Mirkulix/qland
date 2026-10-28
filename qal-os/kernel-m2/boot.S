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

    # ---- GDT: null, ring0 code, ring0 data ----
    lgdt gdt_ptr
    mov $0x10, %ax
    mov %ax, %ds
    mov %ax, %es
    mov %ax, %fs
    mov %ax, %gs
    mov %ax, %ss
    ljmp $0x08, $gdt_flush
gdt_flush:

    # ---- IDT: 256 zeroed entries, timer at 32 ----
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
    lidt idt_ptr

    # ---- PIC remap to 0x20/0x28, only IRQ0 ----
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

    # ---- PIT: 100 Hz ----
    mov $0x36, %al
    out %al, $0x43
    mov $0x9B, %al
    out %al, $0x40
    mov $0x2E, %al
    out %al, $0x40

    # ---- COM1: 115200 8N1 ----
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

    # ---- paging: identity-map first 4 MiB ----
    mov $page_table, %edi
    mov $0x3, %eax
    mov $1024, %ecx
.fill_pt:
    mov %eax, (%edi)
    add $0x1000, %eax
    add $4, %edi
    loop .fill_pt
    mov $page_dir, %edi
    mov $page_table, %eax
    or $0x3, %eax
    mov %eax, (%edi)
    mov $page_dir, %eax
    mov %eax, %cr3
    mov %cr0, %eax
    or $0x80000000, %eax
    mov %eax, %cr0
    jmp paging_on
paging_on:

    # ---- two initial task frames ----
    mov $task_a_stack_top, %eax
    sub $44, %eax
    movl $0, 0(%eax)      # edi
    movl $0, 4(%eax)      # esi
    movl $0, 8(%eax)      # ebp
    movl $0, 12(%eax)     # esp (ignored by popa)
    movl $0, 16(%eax)     # ebx
    movl $0, 20(%eax)     # edx
    movl $0, 24(%eax)     # ecx
    movl $0, 28(%eax)     # eax
    movl $task_a, 32(%eax)
    movl $0x08, 36(%eax)
    movl $0x202, 40(%eax)
    mov %eax, task_a_esp

    mov $task_b_stack_top, %eax
    sub $44, %eax
    movl $0, 0(%eax)
    movl $0, 4(%eax)
    movl $0, 8(%eax)
    movl $0, 12(%eax)
    movl $0, 16(%eax)
    movl $0, 20(%eax)
    movl $0, 24(%eax)
    movl $0, 28(%eax)
    movl $task_b, 32(%eax)
    movl $0x08, 36(%eax)
    movl $0x202, 40(%eax)
    mov %eax, task_b_esp
    mov %eax, next_esp

    mov $boot_msg, %esi
    call serial_print
    sti

idle:
    hlt
    jmp idle

# ---- timer ISR: save, round-robin, EOI, restore ----
irq0:
    pusha
    mov %esp, current_esp
    mov next_esp, %esp
    mov %esp, %eax
    cmp task_a_esp, %eax
    jne to_a
    mov task_b_esp, %eax
    jmp set_next
to_a:
    mov task_a_esp, %eax
set_next:
    mov %eax, next_esp
    mov $0x20, %dx
    mov $0x20, %al
    out %al, %dx
    popa
    iret

# ---- tasks ----
task_a:
    cmpl $0, a_sent
    jne .a_char
    testb $1, task_a_caps
    jz .a_deny
    movl $1, a_sent
    movl $0x51, mailbox
    mov $msg_send_ok, %esi
    call serial_print
    jmp .a_char
.a_deny:
    movl $1, a_sent
    mov $msg_send_deny, %esi
    call serial_print
.a_char:
    mov $msg_a, %esi
    call serial_print
    hlt
    jmp task_a

task_b:
    cmpl $0, b_tried
    jne .b_recv
    movl $1, b_tried
    testb $1, task_b_caps
    jz .b_deny
    mov $msg_b_send_ok, %esi
    call serial_print
    jmp .b_recv
.b_deny:
    mov $msg_b_deny, %esi
    call serial_print
.b_recv:
    cmpl $0x51, mailbox
    jne .b_char
    movl $0, mailbox
    mov $msg_recv, %esi
    call serial_print
.b_char:
    mov $msg_b, %esi
    call serial_print
    hlt
    jmp task_b

# ---- serial print (ESI = asciz) ----
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
    .asciz "QAL-OS M2: paging + capability-gated IPC + scheduler\n"
msg_a:
    .asciz "A"
msg_b:
    .asciz "B"
msg_send_ok:
    .asciz "[A:cap ok -> B]"
msg_send_deny:
    .asciz "[A:cap denied]"
msg_recv:
    .asciz "[B:recv]"
msg_b_send_ok:
    .asciz "[B:cap ok]"
msg_b_deny:
    .asciz "[B:cap denied]"


.section .data
.align 8
gdt:
    .quad 0x0000000000000000
    .quad 0x00CF9A000000FFFF
    .quad 0x00CF92000000FFFF
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
current_esp: .long 0
next_esp:    .long 0
task_a_esp:  .long 0
task_b_esp:  .long 0
task_a_caps: .long 1
task_b_caps: .long 0
a_sent:      .long 0
b_tried:     .long 0
mailbox:     .long 0

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
task_a_stack:
    .skip 4096
task_a_stack_top:
.align 16
task_b_stack:
    .skip 4096
task_b_stack_top:

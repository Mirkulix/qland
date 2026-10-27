CC ?= clang
LD ?= ld.lld
CFLAGS := --target=i386-none-elf -c
LDFLAGS := -m elf_i386 -T linker.ld
all: kernel.elf
boot.o: boot.S
	$(CC) $(CFLAGS) $< -o $@
kernel.elf: boot.o linker.ld
	$(LD) $(LDFLAGS) -o $@ boot.o
run: kernel.elf
	qemu-system-x86_64 -kernel kernel.elf -serial stdio -display none -no-reboot
clean:
	rm -f boot.o kernel.elf serial.log

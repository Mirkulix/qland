CC ?= clang
LD ?= ld.lld
CFLAGS := --target=i386-none-elf -c
CFLAGS_C := --target=i386-none-elf -ffreestanding -fno-builtin -fno-stack-protector -O2 -march=i386 -mno-sse -mno-sse2 -mno-mmx
LDFLAGS := -m elf_i386 -T linker.ld
all: kernel.elf
boot.o: boot.S
	$(CC) $(CFLAGS) $< -o $@
sha256.o: sha256.c
	$(CC) $(CFLAGS_C) $< -o $@
kernel.elf: boot.o sha256.o linker.ld
	$(LD) $(LDFLAGS) -o $@ boot.o sha256.o
run: kernel.elf
	qemu-system-x86_64 -kernel kernel.elf -drive file=disk.img,format=raw,if=ide,index=0 -serial stdio -display none -no-reboot
clean:
	rm -f boot.o sha256.o kernel.elf serial.log

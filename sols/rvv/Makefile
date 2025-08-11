
# make sure to run these commands inside docker environment

build:
	riscv64-unknown-linux-musl-gcc -static -g -O1 -Wall -Wextra -fno-common -fno-builtin-printf -march=rv64gcv -mabi=lp64d -mcmodel=medany -Wl,-Map=output.map -T ./k230.ld dot.S simple.c -o simple.elf

clean:
	rm -f simple.elf
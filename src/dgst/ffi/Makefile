# Makefile for building the shared library libfilters.so.

# Targets.
all: libfilters.so libfilters.so.1.0.0

# Create a symbolic link for the shared library.
libfilters.so: libfilters.so.1.0.0
	ln -sf libfilters.so.1.0.0 libfilters.so

# Compile the C source file into a shared library.
libfilters.so.1.0.0: filters.c filters.h
	clang -std=c23 -Wall -Wextra -Wpedantic -O3 -march=native -mavx -mavx2 -mavx512f -mfma -mtune=native -fPIC -c filters.c -o filters.o
	clang -shared -Wl,-soname,libfilters.so.1 -o libfilters.so.1.0.0 filters.o

# Clean up generated files.
clean:
	rm -f *.o *.so*

# Phony targets.
.PHONY: all clean

# Makefile for building the shared library libfilters.so.

# Targets.
all: libfilters.so libfilters.so.1.0.0

# Create a symbolic link for the shared library.
libfilters.so: libfilters.so.1.0.0
	ln -sf __objs__/libfilters.so.1.0.0 __objs__/libfilters.so

# Compile the C source file into a shared library.
libfilters.so.1.0.0: filters.c filters.h
	@mkdir -p __objs__
	clang -std=c23 -fopenmp=libomp -Wall -Wextra -Wpedantic -O3 -march=native -mavx -mavx2 -mavx512f -mfma -mtune=native -fPIC -c filters.c -o __objs__/filters.o
	clang -shared -Wl,-soname,libfilters.so.1 -fopenmp=libomp -o __objs__/libfilters.so.1.0.0 __objs__/filters.o

# Clean up generated files.
clean:
	rm -f __objs__/*.o __objs__/*.so*

# Phony targets.
.PHONY: all clean

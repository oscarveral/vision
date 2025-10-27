# Makefile for building the shared library libfilters.so.

# Targets.
all: libfilters.so libfilters.so.1.0.0

# Create a symbolic link for the shared library.
libfilters.so: libfilters.so.1.0.0
	ln -sf __objs__/libfilters.so.1.0.0 __objs__/libfilters.so

# Source files.
SRCS := $(wildcard *.c)
HEADERS := $(wildcard *.h)
OBJS := $(patsubst %.c,__objs__/%.o,$(SRCS))

# Compile the C source file into a shared library.
libfilters.so.1.0.0: $(SRCS) $(HEADERS)
	@mkdir -p __objs__
	clang -std=c23 -fopenmp=libomp -Wall -Wextra -Wpedantic -O3 -march=native -mavx -mavx2 -mavx512f -mfma -mtune=native -fPIC -c $(SRCS)
	@mv *.o __objs__/
	clang -shared -Wl,-soname,libfilters.so.1 -fopenmp=libomp -o __objs__/libfilters.so.1.0.0 $(OBJS)

# Clean up generated files.
clean:
	rm -f __objs__/*.o __objs__/*.so*

# Phony targets.
.PHONY: all clean

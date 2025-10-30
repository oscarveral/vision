# Makefile for building the shared library libfilters.so.

# Targets.
all: libfilters.so libfilters.so.1.0.0

# Create a symbolic link for the shared library.
libfilters.so: libfilters.so.1.0.0
	ln -sf __objs__/libfilters.so.1.0.0 __objs__/libfilters.so

# Compiler and flags
CC := clang
CFLAGS := -std=c23 -fopenmp=libomp -Wall -Wextra -Wpedantic -O3 -march=native -mavx -mavx2 -mavx512f -mfma -mtune=native -fPIC
LDFLAGS := -lm

# Source files.
SRCS := $(wildcard *.c)
HEADERS := $(wildcard *.h)
OBJS := $(patsubst %.c,__objs__/%.o,$(SRCS))

# Ensure object directory exists before compiling
__OBJS_DIR := __objs__

.PHONY: dirs
dirs:
	@mkdir -p $(__OBJS_DIR)

# Pattern rule: compile each .c into __objs__/name.o
__objs__/%.o: %.c $(HEADERS) | dirs
	$(CC) $(CFLAGS) -c $< -o $@


# Compile the C source file into a shared library.
libfilters.so.1.0.0: $(OBJS) $(HEADERS)
	@mkdir -p $(__OBJS_DIR)
	$(CC) -shared -Wl,-soname,libfilters.so.1 -fopenmp=libomp -o $(__OBJS_DIR)/libfilters.so.1.0.0 $(OBJS) $(LDFLAGS)

# Clean up generated files.
clean:
	rm -f $(__OBJS_DIR)/*.o $(__OBJS_DIR)/*.so*

# Phony targets.
.PHONY: all clean

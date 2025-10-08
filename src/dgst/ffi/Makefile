

all: libfilters.so libfilters.so.1.0.0

libfilters.so: libfilters.so.1.0.0
	ln -sf libfilters.so.1.0.0 libfilters.so

libfilters.so.1.0.0: filters.c filters.h
	gcc -Wall -c -fPIC filters.c
	gcc -shared -Wl,-soname,libfilters.so.1 -o libfilters.so.1.0.0 filters.o

clean:
	rm -f *.o *.so*

.PHONY: all clean

CC=gcc

install: librml.so
	cp librml.so /usr/lib/librml.so
	cp rml.h /usr/include/rml.h
librml.so: internal.o tensor.o
	$(CC) -shared -Wl,-soname,$@ -o $@ $^
tensor.o: tensor.c rml.h internal.h tensor.h
	$(CC) $< -c -o $@ -fPIC
internal.o: internal.c rml.h internal.h
	$(CC) $< -c -o $@ -fPIC
clean:
	rm *.o librml.so

.PHONY: install

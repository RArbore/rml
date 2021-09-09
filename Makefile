CC=gcc
OBJ_FLAGS=-fPIC
L_FLAGS=-lcblas

install: librml.so
	cp librml.so /usr/lib/librml.so
	cp rml.h /usr/include/rml.h
librml.so: internal.o fileio.o tensor.o
	$(CC) -shared -Wl,-soname,$@ -o $@ $^ $(L_FLAGS)
tensor.o: tensor.c rml.h internal.h tensor.h
	$(CC) $< -c -o $@ $(OBJ_FLAGS)
fileio.o: fileio.c rml.h internal.h fileio.h
	$(CC) $< -c -o $@ $(OBJ_FLAGS)
internal.o: internal.c rml.h internal.h
	$(CC) $< -c -o $@ $(OBJ_FLAGS)
clean:
	rm *.o librml.so

.PHONY: install

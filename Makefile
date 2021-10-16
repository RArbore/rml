#   This file is part of rml. \
\
    rml is free software: you can redistribute it and/or modify \
    it under the terms of the GNU Lesser General Public License as published by \
    the Free Software Foundation, either version 3 of the License, or \
    any later version. \
\
    rml is distributed in the hope that it will be useful, \
    but WITHOUT ANY WARRANTY; without even the implied warranty of \
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the \
    GNU Lesser General Public License for more details. \
\
    You should have received a copy of the GNU Lesser General Public License \
    along with rml. If not, see <https://www.gnu.org/licenses/>.

CC=gcc
OBJ_FLAGS=-fPIC
L_FLAGS=-lcblas -lOpenCL
W_FLAGS=-Wall

install: librml.so
	cp librml.so /usr/lib/librml.so
	cp rml.h /usr/include/rml.h
librml.so: internal.o cl_kernels.o cl_helpers.o fileio.o graph.o grad.o operations.o tensor.o tensor_cl.o tensor_blas.o
	$(CC) -shared -Wl,-soname,$@ -o $@ $^ $(L_FLAGS) $(W_FLAGS)
tensor_blas.o: tensor_blas.c rml.h internal.h tensor_blas.h
	$(CC) $< -c -o $@ $(OBJ_FLAGS)
tensor_cl.o: tensor_cl.c rml.h internal.h tensor_cl.h
	$(CC) $< -c -o $@ $(OBJ_FLAGS)
tensor.o: tensor.c rml.h internal.h tensor.h
	$(CC) $< -c -o $@ $(OBJ_FLAGS)
operations.o: operations.c rml.h internal.h operations.h
	$(CC) $< -c -o $@ $(OBJ_FLAGS)
grad.o: grad.c rml.h grad.h
	$(CC) $< -c -o $@ $(OBJ_FLAGS)
graph.o: graph.c rml.h graph.h
	$(CC) $< -c -o $@ $(OBJ_FLAGS)
fileio.o: fileio.c rml.h internal.h fileio.h
	$(CC) $< -c -o $@ $(OBJ_FLAGS)
cl_helpers.o: cl_helpers.c rml.h cl_helpers.h
	$(CC) $< -c -o $@ $(OBJ_FLAGS)
cl_kernels.o: cl_kernels.c rml.h cl_kernels.h
	$(CC) $< -c -o $@ $(OBJ_FLAGS)
internal.o: internal.c rml.h internal.h
	$(CC) $< -c -o $@ $(OBJ_FLAGS)
clean:
	rm *.o librml.so

.PHONY: install

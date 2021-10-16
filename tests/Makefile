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

test_tensors: test_tensors.o
	$(CC) -o $@ $^ -lrml
test_tensors.o: test_tensors.c
	$(CC) -c -o $@ $^
test_mnist: test_mnist.o
	$(CC) -o $@ $^ -lrml
test_mnist.o: test_mnist.c
	$(CC) -c -o $@ $^
test_cl: test_cl.o
	$(CC) -o $@ $^ -lrml
test_cl.o: test_cl.c
	$(CC) -c -o $@ $^
test_grad: test_grad.o
	$(CC) -o $@ $^ -lrml
test_grad.o: test_grad.c
	$(CC) -c -o $@ $^
clean:
	rm *.o test_tensors test_mnist test_cl test_grad

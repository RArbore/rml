CC=gcc

test_tensors: test_tensors.o
	$(CC) -o $@ $^ -lrml
test_tensors.o: test_tensors.c
	$(CC) -c -o $@ $^
test_mnist: test_mnist.o
	$(CC) -o $@ $^ -lrml
test_mnist.o: test_mnist.c
	$(CC) -c -o $@ $^
clean:
	rm *.o test_tensors test_mnist

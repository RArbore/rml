CC=gcc

exec: test_tensors.o
	$(CC) -o $@ $^ -lrml
test_tensors.o: test_tensors.c
	$(CC) -c -o $@ $^
clean:
	rm *.o exec

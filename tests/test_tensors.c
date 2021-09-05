#include <stdio.h>
#include <time.h>
#include <rml.h>

int main() {
    clock_t t = clock();
    tensor_t *tensor_a = rml_rand_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 1000, 1000));
    tensor_t *tensor_b = rml_rand_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 1000, 1000));
    tensor_t *tensor_c = rml_tensor_matmul_naive(tensor_a, tensor_b);
    t = clock() - t;
    printf("Time taken: %f\n", ((double) t) / CLOCKS_PER_SEC);
}

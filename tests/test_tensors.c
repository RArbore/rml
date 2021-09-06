#include <stdio.h>
#include <time.h>
#include <rml.h>

int main() {
    tensor_t *tensor_a = rml_rand_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 1000, 1000));
    tensor_t *tensor_b = rml_rand_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 1000, 1000));
    clock_t t = clock();
    tensor_t *tensor_c = rml_tensor_matmul_blas(tensor_a, tensor_b);
    t = clock() - t;
    printf("Time taken: %f\n", ((double) t) / CLOCKS_PER_SEC);
    tensor_t *a = rml_create_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 2, 2), 4, 1., 1., 1., 1.);
    tensor_t *b = rml_create_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 2, 3), 6, 1., 2., 3., 4., 5., 6.);
    tensor_t *c = rml_tensor_matmul_blas(a, b);
    rml_print_tensor(c);
}

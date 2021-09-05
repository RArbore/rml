#include <stdio.h>
#include <rml.h>

int main() {
    printf("Hddeloo w\n");
    tensor_t *tensor_a = rml_create_tensor(TENSOR_TYPE_UBYTE, rml_create_dims(2, 2, 2), 4, 1, 1, 1, 1);
    printf("Heloo w\n");
    tensor_t *tensor_b = rml_create_tensor(TENSOR_TYPE_UBYTE, rml_create_dims(2, 2, 3), 6, 1, 2, 3, 4, 5, 6);
    tensor_t *tensor_c = rml_tensor_matmul_naive(tensor_a, tensor_b);
    rml_print_tensor(tensor_c);
}

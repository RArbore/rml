#include <stdio.h>
#include <rml.h>

int main() {
    printf("Hddeloo w\n");
    tensor_t *tensor_a = rml_create_tensor(TENSOR_TYPE_BYTE, rml_create_dims(2, 3, 3), 9, 1, 2, 3, 4, 5, 6, 7, 8, 9);
    printf("Heloo w\n");
    tensor_t *tensor_b = rml_ones_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 3, 3));
    //rml_tensor_mul_inplace(tensor_a, tensor_b);
    rml_print_tensor(tensor_a);
}

#include <stdio.h>
#include <rml.h>

int main() {
    tensor_t *tensor_a = rml_ones_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 3, 3));
    tensor_t *tensor_b = rml_ones_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 3, 3));
    rml_tensor_mul_inplace(tensor_a, tensor_b);
    rml_print_tensor(tensor_a);
}

#include <stdio.h>
#include <time.h>
#include <rml.h>

int main() {
    tensor_t *a = rml_rand_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 3, 3));
    tensor_t *b = rml_rand_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 2, 2));
    rml_print_tensor(a);
    rml_print_dims(a->dims);
    rml_print_tensor(b);
    rml_print_dims(b->dims);
    size_t pos[] = {1, 1};
    tensor_t *c = rml_assign_slice_tensor(a, b, pos);
    rml_print_tensor(c);
    rml_print_dims(c->dims);
}

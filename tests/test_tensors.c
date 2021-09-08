#include <stdio.h>
#include <time.h>
#include <rml.h>

int main() {
    tensor_t *a = rml_rand_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 3, 3));
    rml_print_tensor(a);
    rml_print_dims(a->dims);
    size_t lower[] = {1, 1};
    size_t upper[] = {3, 2};
    tensor_t *b = rml_slice_tensor(a, lower, upper);
    rml_print_tensor(b);
    rml_print_dims(b->dims);
}

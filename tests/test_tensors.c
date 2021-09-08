#include <stdio.h>
#include <time.h>
#include <rml.h>

int main() {
    tensor_t *a = rml_rand_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 3, 3));
    rml_print_tensor(a);
    rml_print_dims(a->dims);
    float c = -1.;
    tensor_t *b = rml_pow_tensor(a, &c);
    rml_print_tensor(b);
    rml_print_dims(b->dims);
    c = -2;
    tensor_t *d = rml_increment_tensor(b, &c);
    rml_print_tensor(d);
    rml_print_dims(d->dims);
    tensor_t *e = rml_abs_tensor(d);
    rml_print_tensor(e);
    rml_print_dims(e->dims);
}

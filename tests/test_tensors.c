#include <stdio.h>
#include <time.h>
#include <rml.h>

int main() {
    tensor_t *a = rml_rand_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 3, 3));
    rml_print_tensor(a);
    rml_print_dims(a->dims);
    size_t pos[] = {0, 1};
    void *prim = rml_primitive_access_tensor(a, pos);
    printf("%f\n", *((float *) prim));
    *((float *) prim) = 10.;
    printf("%f\n", *((float *) prim));
    rml_print_tensor(a);
    rml_print_dims(a->dims);
}

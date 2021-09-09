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
    float *max = rml_max_tensor(e);
    float *min = rml_min_tensor(e);
    printf("%f %f\n", *min, *max);
    tensor_t *softmax = rml_softmax_tensor(e);
    rml_print_tensor(softmax);
    rml_print_dims(softmax->dims);
    float *sum = rml_sum_tensor(softmax);
    printf("%f\n", *sum);
    rml_write_tensor_csv_full("softmax.csv", softmax);
    tensor_t *read_raw = rml_read_tensor_csv_full("softmax.csv");
    rml_print_tensor(read_raw);
    rml_print_dims(read_raw->dims);
}

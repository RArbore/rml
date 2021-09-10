#include <stdio.h>
#include <time.h>
#include <rml.h>

int main() {
    tensor_t *a = rml_rand_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 3, 3));
    rml_print_tensor(a);
    rml_print_dims(a->dims);
    tensor_t *softmax = rml_softmax_tensor(a);
    rml_print_tensor(softmax);
    rml_print_dims(softmax->dims);
    float half = -0.5;
    tensor_t *b = rml_increment_tensor(a, &half);
    rml_print_tensor(b);
    rml_print_dims(b->dims);
    free(softmax);
    softmax = rml_softmax_tensor(b);
    rml_print_tensor(softmax);
    rml_print_dims(softmax->dims);
    tensor_t *relu = rml_relu_tensor(b);
    rml_print_tensor(relu);
    rml_print_dims(relu->dims);
    float point_two = 0.2;
    tensor_t *lrelu = rml_leakyrelu_tensor(b, &point_two);
    rml_print_tensor(lrelu);
    rml_print_dims(lrelu->dims);
    float labels[] = {1, 1, 0, 1, 0, 0, 1, 0, 1};
    tensor_t *label = rml_init_tensor(TENSOR_TYPE_FLOAT, rml_clone_dims(softmax->dims), labels);
    rml_print_tensor(softmax);
    rml_print_tensor(label);
    tensor_t *cross = rml_cross_entropy_loss_tensor(softmax, label);
    rml_print_tensor(cross);
    rml_print_dims(cross->dims);
}

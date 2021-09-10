#include "operations.h"

tensor_t *rml_softmax_tensor(tensor_t *tensor) {
    void *max = rml_max_tensor(tensor);
    SWITCH_ENUM_TYPES(tensor->tensor_type, SCALE_VOID_POINTER_VAL, max, 0, -1);

    tensor_t *decremented = rml_increment_tensor(tensor, max);
    tensor_t *exp = rml_exp_tensor(decremented);
    void *exp_sum = rml_sum_tensor(exp);
    SWITCH_ENUM_TYPES(tensor->tensor_type, INV_VOID_POINTER, exp_sum, 0);
    tensor_t *result = rml_scale_tensor(exp, exp_sum);

    free(max);
    free(decremented);
    free(exp);
    free(exp_sum);

    return result;
}

tensor_t *rml_relu_tensor(tensor_t *tensor) {
    void *zero;
    SWITCH_ENUM_TYPES(tensor->tensor_type, CALLOC_VOID_POINTER, zero, 1);
    tensor_t *result = rml_clamp_tensor(tensor, zero, NULL);
    free(zero);
    return result;
}

tensor_t *rml_leakyrelu_tensor(tensor_t *tensor, void *mult) {
    void *zero;
    SWITCH_ENUM_TYPES(tensor->tensor_type, CALLOC_VOID_POINTER, zero, 1);

    tensor_t *above = rml_clamp_tensor(tensor, zero, NULL);
    tensor_t *below = rml_clamp_tensor(tensor, NULL, zero);
    tensor_t *below_scaled = rml_scale_tensor(below, mult);
    tensor_t *result = rml_add_tensor(above, below_scaled);

    free(zero);
    free(above);
    free(below);
    free(below_scaled);

    return result;
}

tensor_t *rml_cross_entropy_loss_tensor(tensor_t *pred, tensor_t *label) {
    CAST_TENSORS_WIDEN(pred, label);

    void *minus_one;
    SWITCH_ENUM_TYPES(pred->tensor_type, MALLOC_VOID_POINTER, minus_one, 1);
    SWITCH_ENUM_TYPES(pred->tensor_type, ASSIGN_VOID_POINTER, minus_one, -1, 0);

    tensor_t *log_p = rml_log_tensor(pred);
    tensor_t *one = rml_ones_tensor(pred->tensor_type, rml_clone_dims(pred->dims));
    tensor_t *one_minus_p = rml_sub_tensor(one, pred);
    tensor_t *log_one_minus_p = rml_log_tensor(one_minus_p);
    tensor_t *one_minus_label = rml_sub_tensor(one, label);
    tensor_t *term_one = rml_mul_tensor(label, log_p);
    tensor_t *term_two = rml_mul_tensor(one_minus_label, log_one_minus_p);
    tensor_t *added = rml_add_tensor(term_one, term_two);
    tensor_t *result = rml_scale_tensor(added, minus_one);

    free(minus_one);
    free(one);
    free(one_minus_p);
    free(log_one_minus_p);
    free(one_minus_label);
    free(term_one);
    free(term_two);
    free(added);
    CLEANUP_CAST_TENSORS_WIDEN;

    return result;
}

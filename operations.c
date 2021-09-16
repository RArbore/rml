/*  This file is part of rml.

    rml is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    rml is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with rml. If not, see <https://www.gnu.org/licenses/>.  */

#include "operations.h"

tensor_t *rml_softmax_tensor(tensor_t *tensor) {
    void *max = rml_max_tensor(tensor);
    SWITCH_ENUM_TYPES(tensor->tensor_type, SCALE_VOID_POINTER_VAL, max, 0, -1);

    tensor_t *decremented = rml_increment_tensor(tensor, max);
    tensor_t *exp = rml_exp_tensor(decremented);
    tensor_t *exp_sum = rml_sum_tensor(exp);
    SWITCH_ENUM_TYPES(tensor->tensor_type, INV_VOID_POINTER, exp_sum->data, 0);
    tensor_t *result = rml_scale_tensor(exp, exp_sum->data);

    free(max);
    rml_free_tensor(decremented);
    rml_free_tensor(exp);
    rml_free_tensor(exp_sum);

    result->op_code = OP_CODE_SOFTMAX;
    result->source_a = tensor;
    result->source_b = NULL;

    return result;
}

tensor_t *rml_relu_tensor(tensor_t *tensor) {
    void *zero;
    SWITCH_ENUM_TYPES(tensor->tensor_type, CALLOC_VOID_POINTER, zero, 1);
    tensor_t *result = rml_clamp_tensor(tensor, zero, NULL);
    free(zero);

    result->op_code = OP_CODE_RELU;
    result->source_a = tensor;
    result->source_b = NULL;

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
    rml_free_tensor(above);
    rml_free_tensor(below);
    rml_free_tensor(below_scaled);

    result->op_code = OP_CODE_LEAKYRELU;
    result->source_a = tensor;
    result->source_b = NULL;
    result->op_data = malloc(rml_sizeof_type(tensor->tensor_type));
    SWITCH_ENUM_TYPES(tensor->tensor_type, COPY_VOID_POINTER, result->op_data, mult, 0, 0);

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
    rml_free_tensor(one);
    rml_free_tensor(one_minus_p);
    rml_free_tensor(log_one_minus_p);
    rml_free_tensor(one_minus_label);
    rml_free_tensor(term_one);
    rml_free_tensor(term_two);
    rml_free_tensor(added);
    CLEANUP_CAST_TENSORS_WIDEN;

    result->op_code = OP_CODE_CROSSENTROPY;
    result->source_a = pred;
    result->source_b = label;

    return result;
}

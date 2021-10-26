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

    void *minus_one;
    SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, minus_one, 1);
    SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, minus_one, -1, 0);

    tensor_t *decremented = rml_increment_tensor(tensor, max);
    tensor_t *exp = rml_exp_tensor(decremented);
    tensor_t *exp_sum = rml_sum_tensor(exp);
    tensor_t *inv = rml_pow_tensor(exp_sum, minus_one);
    size_t dims[] = {1, 1};
    tensor_t *inv_reshape = rml_reshape_tensor(inv, dims, 2);
    tensor_t *ones = rml_ones_tensor(tensor->tensor_type, rml_create_dims(2, exp->dims->flat_size, 1));
    rml_cl_make_same_device(ones, tensor);
    tensor_t *inv_repeat = rml_matmul_tensor(ones, inv_reshape);
    tensor_t *inv_repeat_reshape = rml_reshape_tensor(inv_repeat, tensor->dims->dims, tensor->dims->num_dims);
    tensor_t *result = rml_mul_tensor(exp, inv_repeat_reshape);

    free(max);
    free(minus_one);

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

    return result;
}

tensor_t *rml_cross_entropy_loss_tensor(tensor_t *pred, tensor_t *label) {
    void *minus_one;
    SWITCH_ENUM_TYPES(pred->tensor_type, MALLOC_VOID_POINTER, minus_one, 1);
    SWITCH_ENUM_TYPES(pred->tensor_type, ASSIGN_VOID_POINTER, minus_one, -1, 0);

    tensor_t *log_p = rml_log_tensor(pred);
    tensor_t *one = rml_ones_tensor(pred->tensor_type, rml_clone_dims(pred->dims));
    rml_cl_make_same_device(one, pred);
    tensor_t *one_minus_p = rml_sub_tensor(one, pred);
    tensor_t *log_one_minus_p = rml_log_tensor(one_minus_p);
    tensor_t *one_minus_label = rml_sub_tensor(one, label);
    tensor_t *term_one = rml_mul_tensor(label, log_p);
    tensor_t *term_two = rml_mul_tensor(one_minus_label, log_one_minus_p);
    tensor_t *added = rml_add_tensor(term_one, term_two);
    tensor_t *result = rml_scale_tensor(added, minus_one);

    free(minus_one);

    return result;
}

tensor_t *rml_cross_entropy_loss_safe_tensor(tensor_t* pred, tensor_t *label) {
    void *mult, *inc;
    SWITCH_ENUM_TYPES(pred->tensor_type, MALLOC_VOID_POINTER, mult, 1);
    SWITCH_ENUM_TYPES(pred->tensor_type, ASSIGN_VOID_POINTER, mult, 0.9999998, 0);
    SWITCH_ENUM_TYPES(pred->tensor_type, MALLOC_VOID_POINTER, inc, 1);
    SWITCH_ENUM_TYPES(pred->tensor_type, ASSIGN_VOID_POINTER, inc, 0.0000001, 0);

    tensor_t *pred_adj1 = rml_scale_tensor(pred, mult);
    tensor_t *pred_adj2 = rml_increment_tensor(pred_adj1, inc);
    tensor_t *result = rml_cross_entropy_loss_tensor(pred_adj2, label);

    free(mult);
    free(inc);

    return result;
}

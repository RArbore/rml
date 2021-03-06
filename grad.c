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

#include "grad.h"

void rml_calc_gradient(tensor_t *tensor) {
    if (tensor == NULL) return;
    switch (tensor->op_code) {
        case OP_CODE_CREATE: {
            tensor->jacob_a = NULL;
            tensor->jacob_b = NULL;
            break;
        }
        case OP_CODE_PARAM: {
            tensor->jacob_a = NULL;
            tensor->jacob_b = NULL;
            break;
        }
        case OP_CODE_CLONE: {
            tensor_t *ones = NULL;
            if (rml_cl_tensor_on_cl(tensor)) ones = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->dims->flat_size));
            else ones = rml_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->dims->flat_size));
            tensor->jacob_a = rml_diag_tensor(ones, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(ones);
            break;
        }
        case OP_CODE_MATMUL: {
            tensor_t *grad_a = NULL, *grad_b = NULL;
            if (rml_cl_tensor_on_cl(tensor)) {
                grad_a = rml_cl_zeros_tensor(tensor->tensor_type, rml_create_dims(2, tensor->dims->flat_size, tensor->source_a->dims->flat_size));
                grad_b = rml_cl_zeros_tensor(tensor->tensor_type, rml_create_dims(2, tensor->dims->flat_size, tensor->source_b->dims->flat_size));
            }
            else {
                grad_a = rml_zeros_tensor(tensor->tensor_type, rml_create_dims(2, tensor->dims->flat_size, tensor->source_a->dims->flat_size));
                grad_b = rml_zeros_tensor(tensor->tensor_type, rml_create_dims(2, tensor->dims->flat_size, tensor->source_b->dims->flat_size));
            }
            tensor_t *b_transpose = rml_transpose_tensor(tensor->source_b);
            for (size_t i = 0; i < tensor->dims->flat_size / b_transpose->dims->dims[0]; i ++) {
                size_t pos[2];
                pos[0] = i * b_transpose->dims->dims[0];
                pos[1] = i * b_transpose->dims->dims[1];
                tensor_t *new_grad_a = rml_assign_slice_tensor(grad_a, b_transpose, pos);
                rml_free_tensor(grad_a);
                grad_a = new_grad_a;
                new_grad_a->source_a = NULL;
            }
            for (size_t i = 0; i < tensor->dims->flat_size / tensor->source_b->dims->dims[1]; i++) {
                for (size_t j = 0; j < tensor->source_b->dims->flat_size / tensor->source_b->dims->dims[1]; j++) {
                    size_t pos[2];
                    pos[0] = i * tensor->source_b->dims->dims[1];
                    pos[1] = j * tensor->source_b->dims->dims[1];
                    size_t lower[2];
                    lower[0] = i;
                    lower[1] = j;
                    size_t upper[2];
                    upper[0] = i + 1;
                    upper[1] = j + 1;
                    tensor_t *scale_constant = rml_slice_tensor(tensor->source_a, lower, upper);
                    tensor_t *ones_vec;
                    if (rml_cl_tensor_on_cl(tensor)) ones_vec = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(2, tensor->source_b->dims->dims[1], 1));
                    else ones_vec = rml_ones_tensor(tensor->tensor_type, rml_create_dims(2, tensor->source_b->dims->dims[1], 1));
                    tensor_t *scalar_repeat = rml_matmul_tensor(ones_vec, scale_constant);
                    tensor_t *scalar_reshape = rml_reshape_tensor(scalar_repeat, &tensor->source_b->dims->dims[1],  1);
                    tensor_t *scalar_diag = rml_diag_tensor(scalar_reshape, 2);
                    tensor_t *new_grad_b = rml_assign_slice_tensor(grad_b, scalar_diag, pos);
                    rml_free_tensor(grad_b);
                    grad_b = new_grad_b;
                    new_grad_b->source_a = NULL;
                    rml_free_tensor(scale_constant);
                    rml_free_tensor(ones_vec);
                    rml_free_tensor(scalar_repeat);
                    rml_free_tensor(scalar_reshape);
                    rml_free_tensor(scalar_diag);
                }
            }
            tensor->jacob_a = grad_a;
            tensor->jacob_b = grad_b;
            rml_free_tensor(b_transpose);
            break;
        }
        case OP_CODE_CONCAT: {
            tensor_t *zeros_a = NULL, *zeros_b = NULL, *ones_a = NULL, *ones_b = NULL, *cat_a = NULL, *cat_b = NULL, *grad_a = NULL, *grad_b = NULL, *one = NULL;
            if (rml_cl_tensor_on_cl(tensor)) {
                zeros_a = rml_zeros_tensor(tensor->tensor_type, rml_clone_dims(tensor->source_a->dims));
                zeros_b = rml_zeros_tensor(tensor->tensor_type, rml_clone_dims(tensor->source_b->dims));
                ones_a = rml_ones_tensor(tensor->tensor_type, rml_clone_dims(tensor->source_a->dims));
                ones_b = rml_ones_tensor(tensor->tensor_type, rml_clone_dims(tensor->source_b->dims));
                cat_a = rml_concat_tensor(ones_a, zeros_a, *((size_t *) tensor->op_data));
                cat_b = rml_concat_tensor(zeros_b, ones_b, *((size_t *) tensor->op_data));
                grad_a = rml_cl_zeros_tensor(tensor->tensor_type, rml_create_dims(2, cat_a->dims->flat_size, tensor->source_a->dims->flat_size));
                grad_b = rml_cl_zeros_tensor(tensor->tensor_type, rml_create_dims(2, cat_b->dims->flat_size, tensor->source_b->dims->flat_size));
                one = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(2, 1, 1));
            }
            else {
                zeros_a = rml_zeros_tensor(tensor->tensor_type, rml_clone_dims(tensor->source_a->dims));
                zeros_b = rml_zeros_tensor(tensor->tensor_type, rml_clone_dims(tensor->source_b->dims));
                ones_a = rml_ones_tensor(tensor->tensor_type, rml_clone_dims(tensor->source_a->dims));
                ones_b = rml_ones_tensor(tensor->tensor_type, rml_clone_dims(tensor->source_b->dims));
                cat_a = rml_concat_tensor(ones_a, zeros_a, *((size_t *) tensor->op_data));
                cat_b = rml_concat_tensor(zeros_b, ones_b, *((size_t *) tensor->op_data));
                grad_a = rml_zeros_tensor(tensor->tensor_type, rml_create_dims(2, cat_a->dims->flat_size, tensor->source_a->dims->flat_size));
                grad_b = rml_zeros_tensor(tensor->tensor_type, rml_create_dims(2, cat_b->dims->flat_size, tensor->source_b->dims->flat_size));
                one = rml_ones_tensor(tensor->tensor_type, rml_create_dims(2, 1, 1));
            }
            for (size_t r = 0, c = 0; r < cat_a->dims->flat_size; r++) {
                int is_zero;
                SWITCH_ENUM_TYPES(tensor->tensor_type, IS_ZERO_VOID_POINTER, cat_a->data, r, is_zero);
                if (is_zero) continue;
                size_t pos[2];
                pos[0] = r;
                pos[1] = c;
                tensor_t *new_grad_a = rml_assign_slice_tensor(grad_a, one, pos);
                rml_free_tensor(grad_a);
                grad_a = new_grad_a;
                new_grad_a->source_a = NULL;
                c++;
            }
            for (size_t r = 0, c = 0; r < cat_b->dims->flat_size; r++) {
                int is_zero;
                SWITCH_ENUM_TYPES(tensor->tensor_type, IS_ZERO_VOID_POINTER, cat_b->data, r, is_zero);
                if (is_zero) continue;
                size_t pos[2];
                pos[0] = r;
                pos[1] = c;
                tensor_t *new_grad_b = rml_assign_slice_tensor(grad_b, one, pos);
                rml_free_tensor(grad_b);
                grad_b = new_grad_b;
                new_grad_b->source_a = NULL;
                c++;
            }
            tensor->jacob_a = grad_a;
            tensor->jacob_b = grad_b;
            rml_free_tensor(zeros_a);
            rml_free_tensor(zeros_b);
            rml_free_tensor(ones_a);
            rml_free_tensor(ones_b);
            rml_free_tensor(cat_a);
            rml_free_tensor(cat_b);
            rml_free_tensor(one);
            break;
        }
        case OP_CODE_SLICE: {
            tensor_t *grad = NULL, *one = NULL;
            if (rml_cl_tensor_on_cl(tensor)) {
                grad = rml_cl_zeros_tensor(tensor->tensor_type, rml_create_dims(2, tensor->dims->flat_size, tensor->source_a->dims->flat_size));
                one = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(2, 1, 1));
            }
            else {
                grad = rml_zeros_tensor(tensor->tensor_type, rml_create_dims(2, tensor->dims->flat_size, tensor->source_a->dims->flat_size));
                one = rml_ones_tensor(tensor->tensor_type, rml_create_dims(2, 1, 1));
            }
            size_t pos_workspace[tensor->dims->num_dims];
            for (size_t i = 0; i < tensor->dims->num_dims; i++) pos_workspace[i] = 0;
            size_t r = 0;
            for (size_t c = 0; c < tensor->source_a->dims->flat_size; c++) {
                int violated = 0;
                for (size_t d = 0; d < tensor->dims->num_dims; d++) {
                    if (pos_workspace[d] < *((size_t *) tensor->op_data + d) || pos_workspace[d] >= *((size_t *) tensor->op_data + d + tensor->dims->num_dims)) {
                        violated = 1;
                        break;
                    }
                }
                if (!violated) {
                    size_t pos[2];
                    pos[0] = r++;
                    pos[1] = c;
                    tensor_t *new_grad = rml_assign_slice_tensor(grad, one, pos);
                    rml_free_tensor(grad);
                    grad = new_grad;
                    new_grad->source_a = NULL;
                }
                pos_workspace[tensor->dims->num_dims - 1]++;
                for (size_t d = tensor->dims->num_dims - 1; d > 0; d--) {
                    if (pos_workspace[d] >= tensor->source_a->dims->dims[d]) {
                        pos_workspace[d] = 0;
                        pos_workspace[d - 1]++;
                    }
                }
            }
            tensor->jacob_a = grad;
            tensor->jacob_b = NULL;
            rml_free_tensor(one);
            break;
        }
        case OP_CODE_ASSIGN_SLICE: {
            tensor_t *grad_a = NULL, *grad_b = NULL, *one = NULL;
            if (rml_cl_tensor_on_cl(tensor)) {
                grad_a = rml_cl_zeros_tensor(tensor->tensor_type, rml_create_dims(2, tensor->dims->flat_size, tensor->source_a->dims->flat_size));
                grad_b = rml_cl_zeros_tensor(tensor->tensor_type, rml_create_dims(2, tensor->dims->flat_size, tensor->source_b->dims->flat_size));
                one = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(2, 1, 1));
            }
            else {
                grad_a = rml_zeros_tensor(tensor->tensor_type, rml_create_dims(2, tensor->dims->flat_size, tensor->source_a->dims->flat_size));
                grad_b = rml_zeros_tensor(tensor->tensor_type, rml_create_dims(2, tensor->dims->flat_size, tensor->source_b->dims->flat_size));
                one = rml_ones_tensor(tensor->tensor_type, rml_create_dims(2, 1, 1));
            }
            size_t pos_workspace[tensor->dims->num_dims];
            for (size_t i = 0; i < tensor->dims->num_dims; i++) pos_workspace[i] = 0;
            size_t c = 0;
            for (size_t i = 0; i < tensor->dims->flat_size; i++) {
                int violated = 0;
                for (size_t d = 0; d < tensor->dims->num_dims; d++) {
                    if (pos_workspace[d] < *((size_t *) tensor->op_data + d) || pos_workspace[d] >= tensor->source_b->dims->dims[d] + *((size_t *) tensor->op_data + d)) {
                        violated = 1;
                        break;
                    }
                }
                if (violated) {
                    size_t pos[2];
                    pos[0] = i;
                    pos[1] = i;
                    tensor_t *new_grad_a = rml_assign_slice_tensor(grad_a, one, pos);
                    rml_free_tensor(grad_a);
                    grad_a = new_grad_a;
                    new_grad_a->source_a = NULL;
                }
                else {
                    size_t pos[2];
                    pos[0] = i;
                    pos[1] = c++;
                    tensor_t *new_grad_b = rml_assign_slice_tensor(grad_b, one, pos);
                    rml_free_tensor(grad_b);
                    grad_b = new_grad_b;
                    new_grad_b->source_a = NULL;
                }
                pos_workspace[tensor->dims->num_dims - 1]++;
                for (size_t d = tensor->dims->num_dims - 1; d > 0; d--) {
                    if (pos_workspace[d] >= tensor->dims->dims[d]) {
                        pos_workspace[d] = 0;
                        pos_workspace[d - 1]++;
                    }
                }
            }
            tensor->jacob_a = grad_a;
            tensor->jacob_b = grad_b;
            rml_free_tensor(one);
            break;
        }
        case OP_CODE_TRANSPOSE: {
            tensor_t *grad = NULL, *one = NULL;
            if (rml_cl_tensor_on_cl(tensor)) {
                grad = rml_cl_zeros_tensor(tensor->tensor_type, rml_create_dims(2, tensor->dims->flat_size, tensor->source_a->dims->flat_size));
                one = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(2, 1, 1));
            }
            else {
                grad = rml_zeros_tensor(tensor->tensor_type, rml_create_dims(2, tensor->dims->flat_size, tensor->source_a->dims->flat_size));
                one = rml_ones_tensor(tensor->tensor_type, rml_create_dims(2, 1, 1));
            }
            for (size_t old_i = 0; old_i < tensor->dims->flat_size; old_i++) {
                size_t r = old_i / tensor->source_a->dims->dims[1];
                size_t c = old_i % tensor->source_a->dims->dims[1];
                size_t new_i = r + c * tensor->dims->dims[1];
                size_t pos[2];
                pos[0] = new_i;
                pos[1] = old_i;
                tensor_t *new_grad = rml_assign_slice_tensor(grad, one, pos);
                rml_free_tensor(grad);
                grad = new_grad;
                new_grad->source_a = NULL;
            }
            tensor->jacob_a = grad;
            tensor->jacob_b = NULL;
            rml_free_tensor(one);
            break;
        }
        case OP_CODE_PERMUTE: {
            tensor_t *grad = NULL, *one = NULL;
            if (rml_cl_tensor_on_cl(tensor)) {
                grad = rml_cl_zeros_tensor(tensor->tensor_type, rml_create_dims(2, tensor->dims->flat_size, tensor->source_a->dims->flat_size));
                one = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(2, 1, 1));
            }
            else {
                grad = rml_zeros_tensor(tensor->tensor_type, rml_create_dims(2, tensor->dims->flat_size, tensor->source_a->dims->flat_size));
                one = rml_ones_tensor(tensor->tensor_type, rml_create_dims(2, 1, 1));
            }
            size_t pos_workspace[tensor->dims->num_dims];
            for (size_t i = 0; i < tensor->dims->num_dims; i++) pos_workspace[i] = 0;
            for (size_t old_i = 0; old_i < tensor->dims->flat_size; old_i++) {
                size_t new_i = 0;
                for (size_t d = 0; d < tensor->dims->num_dims; d++) {
                    size_t prev_mult = 0;
                    if (d > 0) prev_mult = tensor->source_a->dims->dims[*((size_t *) tensor->op_data + d)];
                    new_i = new_i * prev_mult + pos_workspace[*((size_t *) tensor->op_data + d)];
                }
                size_t pos[2];
                pos[0] = new_i;
                pos[1] = old_i;
                tensor_t *new_grad = rml_assign_slice_tensor(grad, one, pos);
                rml_free_tensor(grad);
                grad = new_grad;
                new_grad->source_a = NULL;
                pos_workspace[tensor->dims->num_dims - 1]++;
                for (size_t d = tensor->dims->num_dims - 1; d > 0; d--) {
                    if (pos_workspace[d] >= tensor->source_a->dims->dims[d]) {
                        pos_workspace[d] = 0;
                        pos_workspace[d - 1]++;
                    }
                }
            }
            tensor->jacob_a = grad;
            tensor->jacob_b = NULL;
            rml_free_tensor(one);
            break;
        }
        case OP_CODE_RESHAPE: {
            tensor_t *grad = NULL;
            if (rml_cl_tensor_on_cl(tensor)) grad = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->dims->flat_size));
            else grad = rml_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->dims->flat_size));
            tensor_t *grad_diag = rml_diag_tensor(grad, 2);
            tensor->jacob_a = grad_diag;
            tensor->jacob_b = NULL;
            rml_free_tensor(grad);
            break;
        }
        case OP_CODE_CAST: {
            tensor_t *grad = NULL;
            if (rml_cl_tensor_on_cl(tensor)) grad = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->dims->flat_size));
            else grad = rml_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->dims->flat_size));
            tensor_t *grad_diag = rml_diag_tensor(grad, 2);
            tensor->jacob_a = grad_diag;
            tensor->jacob_b = NULL;
            rml_free_tensor(grad);
            break;
        }
        case OP_CODE_ADD: {
            tensor_t *grad = NULL;
            if (rml_cl_tensor_on_cl(tensor)) grad = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->dims->flat_size));
            else grad = rml_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->dims->flat_size));
            tensor_t *grad_diag = rml_diag_tensor(grad, 2);
            tensor->jacob_a = grad_diag;
            tensor->jacob_b = rml_clone_tensor(grad_diag);
            rml_free_tensor(grad);
            break;
        }
        case OP_CODE_SUB: {
            void *minus_one;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, minus_one, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, minus_one, -1, 0);
            tensor_t *grad = NULL;
            if (rml_cl_tensor_on_cl(tensor)) grad = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->dims->flat_size));
            else grad = rml_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->dims->flat_size));
            tensor_t *grad_scaled = rml_scale_tensor(grad, minus_one);
            tensor->jacob_a = rml_diag_tensor(grad, 2);
            tensor->jacob_b = rml_diag_tensor(grad_scaled, 2);
            rml_free_tensor(grad);
            rml_free_tensor(grad_scaled);
            free(minus_one);
            break;
        }
        case OP_CODE_MUL: {
            tensor_t *source_a_reshape = rml_reshape_tensor(tensor->source_a, &tensor->source_a->dims->flat_size, 1);
            tensor_t *source_b_reshape = rml_reshape_tensor(tensor->source_b, &tensor->source_b->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(source_b_reshape, 2);
            tensor->jacob_b = rml_diag_tensor(source_a_reshape, 2);
            rml_free_tensor(source_a_reshape);
            rml_free_tensor(source_b_reshape);
            break;
        }
        case OP_CODE_DIV: {
            void *minus_one;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, minus_one, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, minus_one, -1, 0);
            void *minus_two;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, minus_two, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, minus_two, -2, 0);
            tensor_t *source_a_reshape = rml_reshape_tensor(tensor->source_a, &tensor->source_a->dims->flat_size, 1);
            tensor_t *source_b_reshape = rml_reshape_tensor(tensor->source_b, &tensor->source_b->dims->flat_size, 1);
            tensor_t *source_a_neg = rml_scale_tensor(source_a_reshape, minus_one);
            tensor_t *source_b_inv_sq = rml_pow_tensor(source_b_reshape, minus_two);
            tensor_t *grad_a = rml_pow_tensor(source_b_reshape, minus_one);
            tensor_t *grad_b = rml_mul_tensor(source_a_neg, source_b_inv_sq);
            tensor->jacob_a = rml_diag_tensor(grad_a, 2);
            tensor->jacob_b = rml_diag_tensor(grad_b, 2);
            rml_free_tensor(source_a_reshape);
            rml_free_tensor(source_b_reshape);
            rml_free_tensor(source_a_neg);
            rml_free_tensor(source_b_inv_sq);
            rml_free_tensor(grad_a);
            rml_free_tensor(grad_b);
            free(minus_one);
            free(minus_two);
            break;
        }
        case OP_CODE_INCREMENT: {
            tensor_t *grad = NULL;
            if (rml_cl_tensor_on_cl(tensor)) grad = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->dims->flat_size));
            else grad = rml_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->dims->flat_size));
            tensor->jacob_a = rml_diag_tensor(grad, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(grad);
            break;
        }
        case OP_CODE_SCALE: {
            tensor_t *grad = NULL;
            if (rml_cl_tensor_on_cl(tensor)) grad = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->dims->flat_size));
            else grad = rml_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->dims->flat_size));
            tensor_t *scaled = rml_scale_tensor(grad, tensor->op_data);
            tensor->jacob_a = rml_diag_tensor(scaled, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(grad);
            rml_free_tensor(scaled);
            break;
        }
        case OP_CODE_EXP: {
            tensor_t *reshape = rml_reshape_tensor(tensor, &tensor->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(reshape, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(reshape);
            break;
        }
        case OP_CODE_LOG: {
            void *minus_one;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, minus_one, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, minus_one, -1, 0);
            tensor_t *pow = rml_pow_tensor(tensor, minus_one);
            tensor_t *reshape = rml_reshape_tensor(pow, &pow->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(reshape, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(pow);
            rml_free_tensor(reshape);
            free(minus_one);
            break;
        }
        case OP_CODE_POW: {
            void *dec_power = malloc(rml_sizeof_type(tensor->tensor_type));
            SWITCH_ENUM_TYPES(tensor->tensor_type, COPY_VOID_POINTER, dec_power, tensor->op_data, 0, 0);
            SWITCH_ENUM_TYPES(tensor->tensor_type, INCREMENT_VOID_POINTER_VAL, dec_power, 0, -1);
            tensor_t *pow = rml_pow_tensor(tensor->source_a, dec_power);
            tensor_t *grad = rml_scale_tensor(pow, tensor->op_data);
            tensor_t *grad_reshaped = rml_reshape_tensor(grad, &tensor->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(grad_reshaped, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(pow);
            rml_free_tensor(grad);
            rml_free_tensor(grad_reshaped);
            free(dec_power);
            break;
        }
        case OP_CODE_SIN: {
            tensor_t *cos = rml_cos_tensor(tensor->source_a);
            tensor_t *reshape = rml_reshape_tensor(cos, &cos->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(reshape, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(cos);
            rml_free_tensor(reshape);
            break;
        }
        case OP_CODE_COS: {
            void *minus_one;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, minus_one, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, minus_one, -1, 0);
            tensor_t *sin = rml_sin_tensor(tensor->source_a);
            tensor_t *scaled = rml_scale_tensor(sin, minus_one);
            tensor_t *reshape = rml_reshape_tensor(scaled, &scaled->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(reshape, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(sin);
            rml_free_tensor(scaled);
            rml_free_tensor(reshape);
            free(minus_one);
            break;
        }
        case OP_CODE_TAN: {
            void *minus_two;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, minus_two, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, minus_two, -2, 0);
            tensor_t *cos = rml_cos_tensor(tensor->source_a);
            tensor_t *inv_sq = rml_pow_tensor(cos, minus_two);
            tensor_t *reshape = rml_reshape_tensor(inv_sq, &inv_sq->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(reshape, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(cos);
            rml_free_tensor(inv_sq);
            rml_free_tensor(reshape);
            free(minus_two);
            break;
        }
        case OP_CODE_SINH: {
            tensor_t *cosh = rml_cosh_tensor(tensor->source_a);
            tensor_t *reshape = rml_reshape_tensor(cosh, &cosh->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(reshape, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(cosh);
            rml_free_tensor(reshape);
            break;
        }
        case OP_CODE_COSH: {
            tensor_t *sinh = rml_sinh_tensor(tensor->source_a);
            tensor_t *reshape = rml_reshape_tensor(sinh, &sinh->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(reshape, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(sinh);
            rml_free_tensor(reshape);
            break;
        }
        case OP_CODE_TANH: {
            tensor_t *sinh = rml_sinh_tensor(tensor->source_a);
            tensor_t *cosh = rml_cosh_tensor(tensor->source_a);
            tensor_t *sinh2 = rml_mul_tensor(sinh, sinh);
            tensor_t *cosh2 = rml_mul_tensor(cosh, cosh);
            tensor_t *numer = rml_sub_tensor(cosh2, sinh2);
            tensor_t *grad = rml_div_tensor(numer, cosh2);
            tensor_t *reshape = rml_reshape_tensor(grad, &grad->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(reshape, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(sinh);
            rml_free_tensor(cosh);
            rml_free_tensor(sinh2);
            rml_free_tensor(cosh2);
            rml_free_tensor(numer);
            rml_free_tensor(grad);
            rml_free_tensor(reshape);
            break;
        }
        case OP_CODE_ASIN: {
            void *minus_one;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, minus_one, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, minus_one, -1, 0);
            void *minus_half;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, minus_half, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, minus_half, -1./2., 0);
            tensor_t *sq = rml_mul_tensor(tensor->source_a, tensor->source_a);
            tensor_t *dec = rml_increment_tensor(sq, minus_one);
            tensor_t *scal = rml_scale_tensor(dec, minus_one);
            tensor_t *grad = rml_pow_tensor(scal, minus_half);
            tensor_t *reshape = rml_reshape_tensor(grad, &grad->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(reshape, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(sq);
            rml_free_tensor(dec);
            rml_free_tensor(scal);
            rml_free_tensor(grad);
            rml_free_tensor(reshape);
            free(minus_one);
            free(minus_half);
            break;
        }
        case OP_CODE_ACOS: {
            void *minus_one;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, minus_one, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, minus_one, -1, 0);
            void *minus_half;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, minus_half, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, minus_half, -1./2., 0);
            tensor_t *sq = rml_mul_tensor(tensor->source_a, tensor->source_a);
            tensor_t *dec = rml_increment_tensor(sq, minus_one);
            tensor_t *scal = rml_scale_tensor(dec, minus_one);
            tensor_t *inv_rt = rml_pow_tensor(scal, minus_half);
            tensor_t *grad = rml_scale_tensor(inv_rt, minus_one);
            tensor_t *reshape = rml_reshape_tensor(grad, &grad->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(reshape, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(sq);
            rml_free_tensor(dec);
            rml_free_tensor(scal);
            rml_free_tensor(inv_rt);
            rml_free_tensor(grad);
            rml_free_tensor(reshape);
            free(minus_one);
            free(minus_half);
            break;
        }
        case OP_CODE_ATAN: {
            void *one;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, one, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, one, 1, 0);
            void *minus_one;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, minus_one, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, minus_one, -1, 0);
            tensor_t *sq = rml_mul_tensor(tensor->source_a, tensor->source_a);
            tensor_t *inc = rml_increment_tensor(sq, one);
            tensor_t *grad = rml_pow_tensor(inc, minus_one);
            tensor_t *reshape = rml_reshape_tensor(grad, &grad->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(reshape, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(sq);
            rml_free_tensor(inc);
            rml_free_tensor(grad);
            rml_free_tensor(reshape);
            free(one);
            free(minus_one);
            break;
        }
        case OP_CODE_ASINH: {
            void *one;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, one, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, one, 1, 0);
            void *minus_half;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, minus_half, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, minus_half, -1./2., 0);
            tensor_t *sq = rml_mul_tensor(tensor->source_a, tensor->source_a);
            tensor_t *inc = rml_increment_tensor(sq, one);
            tensor_t *grad = rml_pow_tensor(inc, minus_half);
            tensor_t *reshape = rml_reshape_tensor(grad, &grad->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(reshape, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(sq);
            rml_free_tensor(inc);
            rml_free_tensor(grad);
            rml_free_tensor(reshape);
            free(one);
            free(minus_half);
            break;
        }
        case OP_CODE_ACOSH: {
            void *minus_one;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, minus_one, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, minus_one, -1, 0);
            void *minus_half;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, minus_half, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, minus_half, -1./2., 0);
            tensor_t *sq = rml_mul_tensor(tensor->source_a, tensor->source_a);
            tensor_t *inc = rml_increment_tensor(sq, minus_one);
            tensor_t *grad = rml_pow_tensor(inc, minus_half);
            tensor_t *reshape = rml_reshape_tensor(grad, &grad->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(reshape, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(sq);
            rml_free_tensor(inc);
            rml_free_tensor(grad);
            rml_free_tensor(reshape);
            free(minus_one);
            free(minus_half);
            break;
        }
        case OP_CODE_ATANH: {
            void *minus_one;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, minus_one, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, minus_one, -1, 0);
            tensor_t *sq = rml_mul_tensor(tensor->source_a, tensor->source_a);
            tensor_t *dec = rml_increment_tensor(sq, minus_one);
            tensor_t *scal = rml_scale_tensor(dec, minus_one);
            tensor_t *grad = rml_pow_tensor(scal, minus_one);
            tensor_t *reshape = rml_reshape_tensor(grad, &grad->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(reshape, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(sq);
            rml_free_tensor(dec);
            rml_free_tensor(scal);
            rml_free_tensor(grad);
            rml_free_tensor(reshape);
            free(minus_one);
            break;
        }
        case OP_CODE_ABS: {
            void *small;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, small, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, small, FLT_EPSILON, 0);
            tensor_t *abs = rml_abs_tensor(tensor->source_a);
            tensor_t *inc = rml_increment_tensor(abs, small);
            tensor_t *grad = rml_div_tensor(tensor->source_a, inc);
            tensor_t *reshape = rml_reshape_tensor(grad, &grad->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(reshape, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(abs);
            rml_free_tensor(inc);
            rml_free_tensor(grad);
            rml_free_tensor(reshape);
            free(small);
            break;
        }
        case OP_CODE_CLAMP: {
            void *small;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, small, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, small, FLT_EPSILON, 0);
            void *minus_one;
            SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, minus_one, 1);
            SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, minus_one, -1, 0);
            tensor_t *diff = rml_sub_tensor(tensor->source_a, tensor);
            tensor_t *inc = rml_increment_tensor(diff, small);
            tensor_t *div = rml_div_tensor(diff, inc);
            tensor_t *inc2 = rml_increment_tensor(div, minus_one);
            tensor_t *grad = rml_scale_tensor(inc2, minus_one);
            tensor_t *reshape = rml_reshape_tensor(grad, &grad->dims->flat_size, 1);
            tensor->jacob_a = rml_diag_tensor(reshape, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(diff);
            rml_free_tensor(inc);
            rml_free_tensor(div);
            rml_free_tensor(inc2);
            rml_free_tensor(grad);
            rml_free_tensor(reshape);
            free(small);
            free(minus_one);
            break;
        }
        case OP_CODE_SUM: {
            if (rml_cl_tensor_on_cl(tensor)) {
                tensor->jacob_a = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(2, 1, tensor->source_a->dims->flat_size));
            }
            else {
                tensor->jacob_a = rml_ones_tensor(tensor->tensor_type, rml_create_dims(2, 1, tensor->source_a->dims->flat_size));
            }
            tensor->jacob_b = NULL;
            break;
        }
        case OP_CODE_DIAG: {
            tensor_t *ones;
            if (rml_cl_tensor_on_cl(tensor)) {
                ones = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->jacob_a->dims->flat_size));
            }
            else {
                ones = rml_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->jacob_a->dims->flat_size));
            }
            tensor_t *diag = rml_diag_tensor(ones, *((size_t *) tensor->op_data) + 1);
            size_t dims[2];
            dims[0] = tensor->dims->flat_size;
            dims[1] = tensor->jacob_a->dims->flat_size;
            tensor->jacob_a = rml_reshape_tensor(diag, dims, 2);
            tensor->jacob_b = NULL;
            rml_free_tensor(ones);
            rml_free_tensor(diag);
            break;
        }
        default: {
            tensor->jacob_a = NULL;
            tensor->jacob_b = NULL;
            printf("Op code #%d doesn't have an associated gradient function.\n", tensor->op_code);
        }
    }
}

int rml_recur_calc_gradients(tensor_t *tensor) {
    if (tensor == NULL) return 0;
    if (tensor->op_code == OP_CODE_CREATE) return 0;
    if (tensor->op_code == OP_CODE_PARAM) return 1;
    int a = rml_recur_calc_gradients(tensor->source_a);
    int b = rml_recur_calc_gradients(tensor->source_b);
    if ((a || b) && tensor->jacob_a == NULL) rml_calc_gradient(tensor);
    return a || b;
}

static gradient_t *rml_merge_gradients(gradient_t *a, gradient_t *b) {
    if (a == NULL && b == NULL) return NULL;
    if (a == NULL) return b;

    while (b != NULL) {
        gradient_t *head = a, *prev = NULL;
        while (head != NULL) {
            if (head->param == b->param) {
                tensor_t *sum = rml_add_tensor(head->grad, b->grad);
                rml_free_tensor(head->grad);
                rml_free_tensor(b->grad);
                head->grad = sum;
                break;
            }
            prev = head;
            head = head->next;
        }
        if (head == NULL) {
            prev->next = b;
            b = b->next;
            prev->next->next = NULL;
        }
        else {
            gradient_t *old_b = b;
            b = b->next;
            free(old_b);
        }
    }
    return a;
}

void rml_free_gradient(gradient_t *grad) {
    if (grad == NULL) return;
    rml_free_tensor(grad->grad);
    rml_free_gradient(grad->next);
    free(grad);
}

static gradient_t *rml_backward_tensor_internal(tensor_t *tensor, tensor_t *acc_grad) {
    if (tensor == NULL || tensor->op_code == OP_CODE_CREATE) return NULL;
    gradient_t *a = NULL;
    gradient_t *b = NULL;
    tensor_t *new_grad_a = NULL;
    tensor_t *new_grad_b = NULL;
    if (tensor->jacob_a != NULL) {
        new_grad_a = rml_matmul_tensor(acc_grad, tensor->jacob_a);
        a = rml_backward_tensor_internal(tensor->source_a, new_grad_a);
    }
    if (tensor->jacob_b != NULL) {
        new_grad_b = rml_matmul_tensor(acc_grad, tensor->jacob_b);
        b = rml_backward_tensor_internal(tensor->source_b, new_grad_b);
    }
    rml_free_tensor(new_grad_a);
    rml_free_tensor(new_grad_b);
    if (tensor->op_code == OP_CODE_PARAM) {
        gradient_t *scratch = malloc(sizeof(gradient_t));
        scratch->param = tensor;
        scratch->grad = rml_clone_tensor(acc_grad);
        scratch->next = NULL;
        return scratch;
    }
    return rml_merge_gradients(a, b);
}

gradient_t *rml_backward_tensor(tensor_t *tensor) {
    assert(tensor->dims->num_dims == 1 && tensor->dims->flat_size == 1);
    rml_recur_calc_gradients(tensor);
    tensor_t *one;
    if (rml_cl_tensor_on_cl(tensor)) {
        one = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(2, 1, 1));
    }
    else {
        one = rml_ones_tensor(tensor->tensor_type, rml_create_dims(2, 1, 1));
    }
    gradient_t *grad = rml_backward_tensor_internal(tensor, one);
    rml_free_tensor(one);
    gradient_t *head = grad;
    while (head != NULL) {
        rml_free_dims(head->grad->dims);
        head->grad->dims = rml_clone_dims(head->param->dims);
        head = head->next;
    }
    return grad;
}

void rml_single_grad_desc_step(gradient_t *grad, void *learning_rate) {
    while (grad != NULL) {
        tensor_t *scaled = rml_scale_tensor(grad->grad, learning_rate);
        rml_sub_tensor_inplace(grad->param, scaled);
        grad = grad->next;
        rml_free_tensor(scaled);
    }
}

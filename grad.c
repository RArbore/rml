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
            tensor_t *identity = rml_diag_tensor(ones, 2);
            tensor->jacob_a = identity;
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
            break;
        }
        case OP_CODE_CONCAT: {
            tensor_t *zeros_a = NULL, *zeros_b = NULL, *ones_a = NULL, *ones_b = NULL, *cat_a = NULL, *cat_b = NULL, *grad_a = NULL, *grad_b = NULL, *one = NULL;
            if (rml_cl_tensor_on_cl(tensor)) {
                zeros_a = rml_cl_zeros_tensor(tensor->tensor_type, rml_clone_dims(tensor->source_a->dims));
                zeros_b = rml_cl_zeros_tensor(tensor->tensor_type, rml_clone_dims(tensor->source_b->dims));
                ones_a = rml_cl_ones_tensor(tensor->tensor_type, rml_clone_dims(tensor->source_a->dims));
                ones_b = rml_cl_ones_tensor(tensor->tensor_type, rml_clone_dims(tensor->source_b->dims));
                cat_a = rml_concat_tensor(ones_a, zeros_a, *((size_t *) tensor->op_data));
                cat_b = rml_concat_tensor(zeros_b, ones_b, *((size_t *) tensor->op_data));
                grad_a = rml_cl_zeros_tensor(tensor->tensor_type, rml_create_dims(2, cat_a->dims->flat_size, tensor->source_a->dims->flat_size));
                grad_b = rml_cl_zeros_tensor(tensor->tensor_type, rml_create_dims(2, cat_b->dims->flat_size, tensor->source_b->dims->flat_size));
                one = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(2, 1, 1));
                rml_print_tensor(cat_a);
                rml_print_tensor(cat_b);
            }
            else {
                zeros_a = rml_cl_zeros_tensor(tensor->tensor_type, rml_clone_dims(tensor->source_a->dims));
                zeros_b = rml_cl_zeros_tensor(tensor->tensor_type, rml_clone_dims(tensor->source_b->dims));
                ones_a = rml_cl_ones_tensor(tensor->tensor_type, rml_clone_dims(tensor->source_a->dims));
                ones_b = rml_cl_ones_tensor(tensor->tensor_type, rml_clone_dims(tensor->source_b->dims));
                cat_a = rml_concat_tensor(ones_a, zeros_a, *((size_t *) tensor->op_data));
                cat_b = rml_concat_tensor(zeros_b, ones_b, *((size_t *) tensor->op_data));
                grad_a = rml_cl_zeros_tensor(tensor->tensor_type, rml_create_dims(2, cat_a->dims->flat_size, tensor->source_a->dims->flat_size));
                grad_b = rml_cl_zeros_tensor(tensor->tensor_type, rml_create_dims(2, cat_b->dims->flat_size, tensor->source_b->dims->flat_size));
                one = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(2, 1, 1));
                rml_print_tensor(cat_a);
                rml_print_tensor(cat_b);
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
            rml_free_tensor(one);
            break;
        }
        default:
            printf("Op code #%d doesn't have an associated gradient function.\n", tensor->op_code);
    }
}
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
        }
        case OP_CODE_PARAM: {
            tensor->jacob_a = NULL;
            tensor->jacob_b = NULL;
        }
        case OP_CODE_CLONE: {
            tensor_t *ones = NULL;
            if (rml_cl_tensor_on_cl(tensor)) ones = rml_cl_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->dims->flat_size));
            else ones = rml_ones_tensor(tensor->tensor_type, rml_create_dims(1, tensor->dims->flat_size));
            tensor_t *identity = rml_diag_tensor(ones, 2);
            tensor->jacob_a = identity;
            tensor->jacob_b = NULL;
            rml_free_tensor(ones);
        }
        case OP_CODE_MATMUL: {
            tensor_t *grad_a = rml_zeros_tensor(tensor->tensor_type, rml_create_dims(2, tensor->dims->flat_size, tensor->source_a->dims->flat_size));
            tensor_t *grad_b = rml_zeros_tensor(tensor->tensor_type, rml_create_dims(2, tensor->dims->flat_size, tensor->source_b->dims->flat_size));
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
                    tensor_t *ones_vec = rml_ones_tensor(tensor->tensor_type, rml_create_dims(2, tensor->source_b->dims->dims[1], 1));
                    tensor_t *scalar_repeat = rml_matmul_tensor(ones_vec, scale_constant);
                    tensor_t *scalar_diag = rml_diag_tensor(scalar_repeat, 2);
                    tensor_t *new_grad_b = rml_assign_slice_tensor(grad_b, scalar_diag, pos);
                    rml_free_tensor(grad_b);
                    grad_b = new_grad_b;
                    new_grad_b->source_a = NULL;
                    rml_free_tensor(scale_constant);
                    rml_free_tensor(ones_vec);
                    rml_free_tensor(scalar_repeat);
                    rml_free_tensor(scalar_diag);
                }
                tensor->jacob_a = grad_a;
                tensor->jacob_b = grad_b;
            }
        }
        default:
            printf("Op code #%d doesn't have an associated gradient function.\n", tensor->op_code);
    }
}

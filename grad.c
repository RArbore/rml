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
        default:
            printf("Op code #%d doesn't have an associated gradient function.\n", tensor->op_code);
    }
}

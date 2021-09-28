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

#ifndef TENSOR_CL_H_
#define TENSOR_CL_H_

#include "cl_helpers.h"
#include "internal.h"
#include "tensor.h"
#include "rml.h"

tensor_t *rml_cl_init_tensor(tensor_type_t type, dims_t *dims, void *data);

tensor_t *rml_cl_zeros_tensor(tensor_type_t type, dims_t *dims);

tensor_t *rml_cl_ones_tensor(tensor_type_t type, dims_t *dims);

tensor_t *rml_cl_clone_tensor(tensor_t *tensor);

tensor_t *rml_cl_matmul_tensor(tensor_t *a, tensor_t *b);

tensor_t *rml_cl_concat_tensor(tensor_t *a, tensor_t *b, size_t dim);

tensor_t *rml_cl_slice_tensor(tensor_t *tensor, size_t *lower_bound, size_t *upper_bound);

tensor_t *rml_cl_assign_slice_tensor(tensor_t *a, tensor_t *b, size_t *lower_bound);

tensor_t *rml_cl_transpose_tensor(tensor_t *tensor);

tensor_t *rml_cl_permute_tensor(tensor_t *tensor, size_t *perms);

tensor_t *rml_cl_cast_float_tensor(tensor_t *tensor);

tensor_t *rml_cl_cast_double_tensor(tensor_t *tensor);

tensor_t *rml_cl_add_tensor(tensor_t *a, tensor_t *b);

tensor_t *rml_cl_sub_tensor(tensor_t *a, tensor_t *b);

tensor_t *rml_cl_mul_tensor(tensor_t *a, tensor_t *b);

tensor_t *rml_cl_div_tensor(tensor_t *a, tensor_t *b);

tensor_t *rml_cl_increment_tensor(tensor_t *a, void *scalar);

tensor_t *rml_cl_scale_tensor(tensor_t *a, void *scalar);

tensor_t *rml_cl_floating_point_op_tensor(tensor_t *tensor, unsigned int cl_op, op_code_t op_code);

tensor_t *rml_cl_clamp_tensor(tensor_t *tensor, void *min, void *max);

void *rml_cl_max_tensor(tensor_t *tensor);

void *rml_cl_min_tensor(tensor_t *tensor);

#endif // TENSOR_CL_H_

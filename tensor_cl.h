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

tensor_t *rml_cl_cast_float_tensor(tensor_t *tensor);

tensor_t *rml_cl_cast_double_tensor(tensor_t *tensor);

#endif // TENSOR_CL_H_

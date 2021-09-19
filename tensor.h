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

#ifndef TENSOR_H_
#define TENSOR_H_

#define CL_TARGET_OPENCL_VERSION 300

#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <assert.h>

#include "tensor_blas.h"
#include "cl_helpers.h"
#include "internal.h"
#include "rml.h"

tensor_t *rml_floating_point_op_tensor(tensor_t *tensor, float (*f)(float), double (*d)(double), long double (*ld)(long double));

size_t rml_sizeof_type(tensor_type_t tensor_type);

#endif // TENSOR_H_

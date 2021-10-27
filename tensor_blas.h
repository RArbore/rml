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

#ifndef TENSOR_BLAS_H_
#define TENSOR_BLAS_H_

#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <assert.h>
#include <cblas.h>

#include "internal.h"
#include "tensor.h"
#include "rml.h"

tensor_t *rml_blas_clone_tensor(tensor_t *tensor);

tensor_t *rml_blas_matmul_tensor(tensor_t *a, tensor_t *b);

tensor_t *rml_blas_add_tensor(tensor_t *a, tensor_t *b);

tensor_t *rml_blas_sub_tensor(tensor_t *a, tensor_t *b);

tensor_t *rml_blas_scale_tensor(tensor_t *a, void *scalar);

void rml_blas_sub_tensor_inplace(tensor_t *a, tensor_t *b);

#endif // TENSOR_BLAS_H_

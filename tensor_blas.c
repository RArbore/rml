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

#include "tensor_blas.h"

tensor_t *rml_matmul_blas_tensor(tensor_t *a, tensor_t *b){
    assert(a->dims->num_dims == 2 && b->dims->num_dims == 2);
    assert(a->dims->dims[1] == b->dims->dims[0]);
    CAST_TENSORS_WIDEN(a, b);
    assert(a->tensor_type == TENSOR_TYPE_FLOAT || a->tensor_type == TENSOR_TYPE_DOUBLE);

    tensor_t *result = rml_zeros_tensor(a->tensor_type, rml_create_dims(2, a->dims->dims[0], b->dims->dims[1]));
    if (result->tensor_type == TENSOR_TYPE_FLOAT) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a->dims->dims[0], b->dims->dims[1], a->dims->dims[1], 1., (float *) a->data, a->dims->dims[1], (float *) b->data, b->dims->dims[1], 0., (float *) result->data, b->dims->dims[1]); \
    }
    else {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a->dims->dims[0], b->dims->dims[1], a->dims->dims[1], 1., (double *) a->data, a->dims->dims[1], (double *) b->data, b->dims->dims[1], 0., (double *) result->data, b->dims->dims[1]); \
    }
    CLEANUP_CAST_TENSORS_WIDEN;

    return result;
}

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

tensor_t *rml_blas_clone_tensor(tensor_t *tensor) {
    assert(tensor->tensor_type == TENSOR_TYPE_FLOAT || tensor->tensor_type == TENSOR_TYPE_DOUBLE);

    tensor_t *clone = rml_init_tensor(tensor->tensor_type, rml_clone_dims(tensor->dims), NULL);
    clone->op_code = OP_CODE_CLONE;
    clone->source_a = tensor;
    if (clone->tensor_type == TENSOR_TYPE_FLOAT) {
        cblas_scopy(clone->dims->flat_size, (float *) tensor->data, 1, (float *) clone->data, 1);
    }
    else {
        cblas_dcopy(clone->dims->flat_size, (double *) tensor->data, 1, (double *) clone->data, 1);
    }

    return clone;
}

tensor_t *rml_blas_matmul_tensor(tensor_t *a, tensor_t *b) {
    assert(a->dims->num_dims == 2 && b->dims->num_dims == 2);
    assert(a->dims->dims[1] == b->dims->dims[0]);
    tensor_t *a_orig = a, *b_orig = b;
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
    result->op_code = OP_CODE_MATMUL;
    result->source_a = a_orig;
    result->source_b = b_orig;

    return result;
}

tensor_t *rml_blas_add_tensor(tensor_t *a, tensor_t *b) {
    tensor_t *a_orig = a, *b_orig = b;
    CAST_TENSORS_WIDEN(a, b)
    assert(rml_dims_equiv(a->dims, b->dims));
    assert(a->tensor_type == TENSOR_TYPE_FLOAT || a->tensor_type == TENSOR_TYPE_DOUBLE);

    tensor_t *result = rml_blas_clone_tensor(a);
    if (result->tensor_type == TENSOR_TYPE_FLOAT) {
        cblas_saxpy(result->dims->flat_size, 1., (float *) b->data, 1, (float *) result->data, 1);
    }
    else {
        cblas_daxpy(result->dims->flat_size, 1., (double *) b->data, 1, (double *) result->data, 1);
    }
    CLEANUP_CAST_TENSORS_WIDEN;
    result->op_code = OP_CODE_ADD;
    result->source_a = a_orig;
    result->source_b = b_orig;

    return result;
}

tensor_t *rml_blas_sub_tensor(tensor_t *a, tensor_t *b) {
    tensor_t *a_orig = a, *b_orig = b;
    CAST_TENSORS_WIDEN(a, b)
    assert(rml_dims_equiv(a->dims, b->dims));
    assert(a->tensor_type == TENSOR_TYPE_FLOAT || a->tensor_type == TENSOR_TYPE_DOUBLE);

    tensor_t *result = rml_blas_clone_tensor(a);
    if (result->tensor_type == TENSOR_TYPE_FLOAT) {
        cblas_saxpy(result->dims->flat_size, -1., (float *) b->data, 1, (float *) result->data, 1);
    }
    else {
        cblas_daxpy(result->dims->flat_size, -1., (double *) b->data, 1, (double *) result->data, 1);
    }
    CLEANUP_CAST_TENSORS_WIDEN;
    result->op_code = OP_CODE_SUB;
    result->source_a = a_orig;
    result->source_b = b_orig;

    return result;
}

tensor_t *rml_blas_scale_tensor(tensor_t *a, void *scalar) {
    assert(a->tensor_type == TENSOR_TYPE_FLOAT || a->tensor_type == TENSOR_TYPE_DOUBLE);

    tensor_t *result = rml_blas_clone_tensor(a);
    if (result->tensor_type == TENSOR_TYPE_FLOAT) {
        cblas_sscal(result->dims->flat_size, *((float *) scalar), (float *) result->data, 1);
    }
    else {
        cblas_dscal(result->dims->flat_size, *((double *) scalar), (double *) result->data, 1);
    }
    result->op_code = OP_CODE_SCALE;
    result->source_a = a;
    result->op_data = malloc(rml_sizeof_type(a->tensor_type));
    SWITCH_ENUM_TYPES(a->tensor_type, COPY_VOID_POINTER, result->op_data, scalar, 0, 0);

    return result;
}

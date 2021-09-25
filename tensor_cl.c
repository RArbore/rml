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

#include "tensor_cl.h"

tensor_t *rml_cl_init_tensor(tensor_type_t type, dims_t *dims, void *data) {
    tensor_t *tensor = malloc(sizeof(tensor_t));

    tensor->tensor_type = type;
    tensor->dims = dims;
    tensor->data = NULL;
    tensor->op_code = OP_CODE_CREATE;
    tensor->source_a = NULL;
    tensor->source_b = NULL;
    tensor->op_data = NULL;

    tensor->cl_mem = malloc(sizeof(cl_mem));
    *((cl_mem *) tensor->cl_mem) = rml_cl_create_buffer(CL_MEM_READ_WRITE, dims->flat_size * rml_sizeof_type(type));
    if (data != NULL) rml_cl_enqueue_write_buffer(*((cl_mem *) tensor->cl_mem), dims->flat_size * rml_sizeof_type(type), data);

    return tensor;
}

tensor_t *rml_cl_zeros_tensor(tensor_type_t type, dims_t *dims) {
    tensor_t *tensor = rml_cl_init_tensor(type, dims, NULL);
    void *zero;
    SWITCH_ENUM_TYPES(tensor->tensor_type, CALLOC_VOID_POINTER, zero, 1);
    rml_cl_enqueue_fill_buffer(*((cl_mem *) tensor->cl_mem), zero, rml_sizeof_type(type), dims->flat_size * rml_sizeof_type(type));

    return tensor;
}

tensor_t *rml_cl_ones_tensor(tensor_type_t type, dims_t *dims) {
    tensor_t *tensor = rml_cl_init_tensor(type, dims, NULL);
    void *one;
    SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, one, 1);
    SWITCH_ENUM_TYPES(tensor->tensor_type, ASSIGN_VOID_POINTER, one, 1, 0);
    rml_cl_enqueue_fill_buffer(*((cl_mem *) tensor->cl_mem), one, rml_sizeof_type(type), dims->flat_size * rml_sizeof_type(type));

    return tensor;
}

tensor_t *rml_cl_clone_tensor(tensor_t *tensor) {
    tensor_t *clone = rml_cl_init_tensor(tensor->tensor_type, rml_clone_dims(tensor->dims), NULL);
    clone->op_code = OP_CODE_CLONE;
    clone->source_a = tensor;

    rml_cl_enqueue_clone_buffer(*((cl_mem *) tensor->cl_mem), *((cl_mem *) clone->cl_mem), clone->dims->flat_size * rml_sizeof_type(clone->tensor_type));

    return clone;
}

tensor_t *rml_cl_matmul_tensor(tensor_t *a, tensor_t *b) {
    tensor_t *a_orig = a, *b_orig = b;
    CAST_TENSORS_WIDEN(a, b);
    tensor_t *result = rml_cl_init_tensor(a->tensor_type, rml_create_dims(2, a->dims->dims[0], b->dims->dims[1]), NULL);
    unsigned int d1 = a->dims->dims[0];
    unsigned int d2 = a->dims->dims[1];
    unsigned int d3 = b->dims->dims[1];

    rml_cl_set_kernel_arg(CL_OP_MATMUL, rml_cl_typeof_tensor(a), 0, a->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_MATMUL, rml_cl_typeof_tensor(a), 1, b->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_MATMUL, rml_cl_typeof_tensor(a), 2, result->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_MATMUL, rml_cl_typeof_tensor(a), 3, &d1, sizeof(unsigned int));
    rml_cl_set_kernel_arg(CL_OP_MATMUL, rml_cl_typeof_tensor(a), 4, &d2, sizeof(unsigned int));
    rml_cl_set_kernel_arg(CL_OP_MATMUL, rml_cl_typeof_tensor(a), 5, &d3, sizeof(unsigned int));
    rml_cl_enqueue_range_kernel(CL_OP_MATMUL, rml_cl_typeof_tensor(a), &result->dims->flat_size);

    result->op_code = OP_CODE_MATMUL;
    result->source_a = a_orig;
    result->source_b = b_orig;
    CLEANUP_CAST_TENSORS_WIDEN;

    return result;
}

tensor_t *rml_cl_concat_tensor(tensor_t *a, tensor_t *b, size_t dim) {
    tensor_t *a_orig = a, *b_orig = b;
    CAST_TENSORS_WIDEN(a, b);
    dims_t *dims = rml_clone_dims(a->dims);
    dims->flat_size /= dims->dims[dim];
    dims->dims[dim] = a->dims->dims[dim] + b->dims->dims[dim];
    dims->flat_size *= dims->dims[dim];
    tensor_t *result = rml_cl_init_tensor(a->tensor_type, dims, NULL);

    unsigned int *dims_a = malloc(a->dims->num_dims * sizeof(unsigned int));
    unsigned int *dims_b = malloc(b->dims->num_dims * sizeof(unsigned int));
    unsigned int *dims_c = malloc(b->dims->num_dims * sizeof(unsigned int));
    for (size_t i = 0; i < a->dims->num_dims; i++) {
        dims_a[i] = (unsigned int) a->dims->dims[i];
        dims_b[i] = (unsigned int) b->dims->dims[i];
        dims_c[i] = (unsigned int) result->dims->dims[i];
    }
    unsigned int num_dims = (unsigned int) a->dims->num_dims;
    unsigned int cdim = (unsigned int) dim;
    cl_mem dims_a_cl = rml_cl_create_buffer(CL_MEM_READ_ONLY, a->dims->num_dims * sizeof(unsigned int));
    cl_mem dims_b_cl = rml_cl_create_buffer(CL_MEM_READ_ONLY, b->dims->num_dims * sizeof(unsigned int));
    cl_mem dims_c_cl = rml_cl_create_buffer(CL_MEM_READ_ONLY, result->dims->num_dims * sizeof(unsigned int));
    rml_cl_enqueue_write_buffer(dims_a_cl, a->dims->num_dims * sizeof(unsigned int), dims_a);
    rml_cl_enqueue_write_buffer(dims_b_cl, b->dims->num_dims * sizeof(unsigned int), dims_b);
    rml_cl_enqueue_write_buffer(dims_c_cl, result->dims->num_dims * sizeof(unsigned int), dims_c);

    rml_cl_set_kernel_arg(CL_OP_CONCAT, rml_cl_typeof_tensor(a), 0, a->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_CONCAT, rml_cl_typeof_tensor(a), 1, b->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_CONCAT, rml_cl_typeof_tensor(a), 2, result->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_CONCAT, rml_cl_typeof_tensor(a), 3, &dims_a_cl, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_CONCAT, rml_cl_typeof_tensor(a), 4, &dims_b_cl, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_CONCAT, rml_cl_typeof_tensor(a), 5, &dims_c_cl, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_CONCAT, rml_cl_typeof_tensor(a), 6, &num_dims, sizeof(unsigned int));
    rml_cl_set_kernel_arg(CL_OP_CONCAT, rml_cl_typeof_tensor(a), 7, &cdim, sizeof(unsigned int));
    rml_cl_enqueue_range_kernel(CL_OP_CONCAT, rml_cl_typeof_tensor(a), &result->dims->flat_size);

    rml_cl_free_buffer(dims_a_cl);
    rml_cl_free_buffer(dims_b_cl);
    rml_cl_free_buffer(dims_c_cl);
    free(dims_a);
    free(dims_b);
    result->op_code = OP_CODE_CONCAT;
    result->source_a = a_orig;
    result->source_b = b_orig;
    CLEANUP_CAST_TENSORS_WIDEN;

    return result;
}

tensor_t *rml_cl_cast_float_tensor(tensor_t *tensor) {
    tensor_t *result = rml_cl_init_tensor(TENSOR_TYPE_FLOAT, rml_clone_dims(tensor->dims), NULL);

    rml_cl_set_kernel_arg(CL_OP_CAST_FLOAT, rml_cl_typeof_tensor(tensor), 0, tensor->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_CAST_FLOAT, rml_cl_typeof_tensor(tensor), 1, result->cl_mem, sizeof(cl_mem));
    rml_cl_enqueue_range_kernel(CL_OP_CAST_FLOAT, rml_cl_typeof_tensor(tensor), &result->dims->flat_size);

    result->op_code = OP_CODE_CAST;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_cl_cast_double_tensor(tensor_t *tensor) {
    tensor_t *result = rml_cl_init_tensor(TENSOR_TYPE_DOUBLE, rml_clone_dims(tensor->dims), NULL);

    rml_cl_set_kernel_arg(CL_OP_CAST_DOUBLE, rml_cl_typeof_tensor(tensor), 0, tensor->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_CAST_DOUBLE, rml_cl_typeof_tensor(tensor), 1, result->cl_mem, sizeof(cl_mem));
    rml_cl_enqueue_range_kernel(CL_OP_CAST_DOUBLE, rml_cl_typeof_tensor(tensor), &result->dims->flat_size);

    result->op_code = OP_CODE_CAST;
    result->source_a = tensor;

    return result;
}

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
    free(dims_c);
    result->op_code = OP_CODE_CONCAT;
    result->source_a = a_orig;
    result->source_b = b_orig;
    CLEANUP_CAST_TENSORS_WIDEN;

    return result;
}

tensor_t *rml_cl_slice_tensor(tensor_t *tensor, size_t *lower_bound, size_t *upper_bound) {
    dims_t *dims = malloc(sizeof(dims_t));
    dims->num_dims = tensor->dims->num_dims;
    dims->dims = malloc(dims->num_dims * sizeof(size_t));
    dims->flat_size = 1;
    for (size_t i = 0; i < tensor->dims->num_dims; i++) {
        assert(lower_bound[i] < upper_bound[i]);
        dims->dims[i] = upper_bound[i] - lower_bound[i];
        dims->flat_size *= dims->dims[i];
    }
    tensor_t *result = rml_cl_init_tensor(tensor->tensor_type, dims, NULL);

    unsigned int *lower_bound_u = malloc(tensor->dims->num_dims * sizeof(unsigned int));
    unsigned int *upper_bound_u = malloc(tensor->dims->num_dims * sizeof(unsigned int));
    unsigned int *tensor_dims = malloc(tensor->dims->num_dims * sizeof(unsigned int));
    unsigned int *result_dims = malloc(tensor->dims->num_dims * sizeof(unsigned int));
    for (size_t i = 0; i < tensor->dims->num_dims; i++) {
        lower_bound_u[i] = (unsigned int) lower_bound[i];
        upper_bound_u[i] = (unsigned int) upper_bound[i];
        tensor_dims[i] = (unsigned int) tensor->dims->dims[i];
        result_dims[i] = (unsigned int) result->dims->dims[i];
    }
    unsigned int num_dims = (unsigned int) tensor->dims->num_dims;
    cl_mem lower_bound_u_cl = rml_cl_create_buffer(CL_MEM_READ_ONLY, tensor->dims->num_dims * sizeof(unsigned int));
    cl_mem upper_bound_u_cl = rml_cl_create_buffer(CL_MEM_READ_ONLY, tensor->dims->num_dims * sizeof(unsigned int));
    cl_mem tensor_dims_cl = rml_cl_create_buffer(CL_MEM_READ_ONLY, tensor->dims->num_dims * sizeof(unsigned int));
    cl_mem result_dims_cl = rml_cl_create_buffer(CL_MEM_READ_ONLY, tensor->dims->num_dims * sizeof(unsigned int));
    rml_cl_enqueue_write_buffer(lower_bound_u_cl, tensor->dims->num_dims * sizeof(unsigned int), lower_bound_u);
    rml_cl_enqueue_write_buffer(upper_bound_u_cl, tensor->dims->num_dims * sizeof(unsigned int), upper_bound_u);
    rml_cl_enqueue_write_buffer(tensor_dims_cl, tensor->dims->num_dims * sizeof(unsigned int), tensor_dims_cl);
    rml_cl_enqueue_write_buffer(result_dims_cl, tensor->dims->num_dims * sizeof(unsigned int), result_dims_cl);

    rml_cl_set_kernel_arg(CL_OP_SLICE, rml_cl_typeof_tensor(tensor), 0, tensor->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_SLICE, rml_cl_typeof_tensor(tensor), 1, result->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_SLICE, rml_cl_typeof_tensor(tensor), 2, &lower_bound_u_cl, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_SLICE, rml_cl_typeof_tensor(tensor), 3, &upper_bound_u_cl, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_SLICE, rml_cl_typeof_tensor(tensor), 4, &tensor_dims_cl, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_SLICE, rml_cl_typeof_tensor(tensor), 5, &result_dims_cl, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_SLICE, rml_cl_typeof_tensor(tensor), 6, &num_dims, sizeof(unsigned int));
    rml_cl_enqueue_range_kernel(CL_OP_SLICE, rml_cl_typeof_tensor(tensor), &result->dims->flat_size);

    rml_cl_free_buffer(lower_bound_u_cl);
    rml_cl_free_buffer(upper_bound_u_cl);
    rml_cl_free_buffer(tensor_dims_cl);
    rml_cl_free_buffer(result_dims_cl);
    free(lower_bound_u);
    free(upper_bound_u);
    free(tensor_dims);
    free(result_dims);
    result->op_code = OP_CODE_SLICE;
    result->source_a = tensor;
    result->op_data = malloc(2 * tensor->dims->num_dims * sizeof(size_t));
    for (size_t i = 0; i < tensor->dims->num_dims; i++) {
        *((size_t *) result->op_data + i) = lower_bound[i];
        *((size_t *) result->op_data + i + tensor->dims->num_dims) = upper_bound[i + tensor->dims->num_dims];
    }

    return result;
}

tensor_t *rml_cl_assign_slice_tensor(tensor_t *a, tensor_t *b, size_t *lower_bound) {
    tensor_t *a_orig = a, *b_orig = b;
    CAST_TENSORS_WIDEN(a, b);
    tensor_t *result = rml_clone_tensor(a);

    unsigned int *lower_bound_u = malloc(result->dims->num_dims * sizeof(unsigned int));
    unsigned int *assign_dims = malloc(result->dims->num_dims * sizeof(unsigned int));
    unsigned int *result_dims = malloc(result->dims->num_dims * sizeof(unsigned int));
    for (size_t i = 0; i < result->dims->num_dims; i++) {
        lower_bound_u[i] = (unsigned int) lower_bound[i];
        assign_dims[i] = (unsigned int) b->dims->dims[i];
        result_dims[i] = (unsigned int) result->dims->dims[i];
    }
    unsigned int num_dims = (unsigned int) result->dims->num_dims;
    cl_mem lower_bound_u_cl = rml_cl_create_buffer(CL_MEM_READ_ONLY, result->dims->num_dims * sizeof(unsigned int));
    cl_mem assign_dims_cl = rml_cl_create_buffer(CL_MEM_READ_ONLY, result->dims->num_dims * sizeof(unsigned int));
    cl_mem result_dims_cl = rml_cl_create_buffer(CL_MEM_READ_ONLY, result->dims->num_dims * sizeof(unsigned int));
    rml_cl_enqueue_write_buffer(lower_bound_u_cl, result->dims->num_dims * sizeof(unsigned int), lower_bound_u);
    rml_cl_enqueue_write_buffer(assign_dims_cl, result->dims->num_dims * sizeof(unsigned int), assign_dims);
    rml_cl_enqueue_write_buffer(result_dims_cl, result->dims->num_dims * sizeof(unsigned int), result_dims);

    rml_cl_set_kernel_arg(CL_OP_ASSIGN_SLICE, rml_cl_typeof_tensor(result), 0, b->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_ASSIGN_SLICE, rml_cl_typeof_tensor(result), 1, result->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_ASSIGN_SLICE, rml_cl_typeof_tensor(result), 2, &lower_bound_u_cl, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_ASSIGN_SLICE, rml_cl_typeof_tensor(result), 3, &assign_dims_cl, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_ASSIGN_SLICE, rml_cl_typeof_tensor(result), 4, &result_dims_cl, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_ASSIGN_SLICE, rml_cl_typeof_tensor(result), 5, &num_dims, sizeof(unsigned int));
    rml_cl_enqueue_range_kernel(CL_OP_ASSIGN_SLICE, rml_cl_typeof_tensor(result), &b->dims->flat_size);

    rml_cl_free_buffer(lower_bound_u_cl);
    rml_cl_free_buffer(assign_dims_cl);
    rml_cl_free_buffer(result_dims_cl);
    free(lower_bound_u);
    free(assign_dims);
    free(result_dims);
    CLEANUP_CAST_TENSORS_WIDEN;
    result->op_code = OP_CODE_ASSIGN_SLICE;
    result->source_a = a_orig;
    result->source_b = b_orig;
    result->op_data = malloc(b->dims->num_dims * sizeof(size_t));
    for (size_t i = 0; i < b->dims->num_dims; i++) {
        *((size_t *) result->op_data) = lower_bound[i];
    }

    return result;
}

tensor_t *rml_cl_transpose_tensor(tensor_t *tensor) {
    tensor_t *result = rml_cl_init_tensor(tensor->tensor_type, rml_clone_dims(tensor->dims), NULL);

    unsigned int in_r = (size_t) tensor->dims->dims[0];
    unsigned int in_c = (size_t) tensor->dims->dims[1];

    rml_cl_set_kernel_arg(CL_OP_TRANSPOSE, rml_cl_typeof_tensor(tensor), 0, tensor->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_TRANSPOSE, rml_cl_typeof_tensor(tensor), 1, result->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_TRANSPOSE, rml_cl_typeof_tensor(tensor), 2, &in_r, sizeof(unsigned int));
    rml_cl_set_kernel_arg(CL_OP_TRANSPOSE, rml_cl_typeof_tensor(tensor), 3, &in_c, sizeof(unsigned int));
    rml_cl_enqueue_range_kernel(CL_OP_TRANSPOSE, rml_cl_typeof_tensor(tensor), &result->dims->flat_size);

    size_t swap = result->dims->dims[0];
    result->dims->dims[0] = result->dims->dims[1];
    result->dims->dims[1] = swap;
    result->op_code = OP_CODE_TRANSPOSE;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_cl_permute_tensor(tensor_t *tensor, size_t *perms) {
    size_t *new_dims = malloc(tensor->dims->num_dims * sizeof(size_t));
    for (size_t i = 0; i < tensor->dims->num_dims; i++) {
        new_dims[i] = tensor->dims->dims[perms[i]];
    }
    dims_t *new_dims_struct = rml_clone_dims(tensor->dims);
    free(new_dims_struct->dims);
    new_dims_struct->dims = new_dims;
    tensor_t *result = rml_cl_init_tensor(tensor->tensor_type, new_dims_struct, NULL);

    unsigned int *perms_u = malloc(tensor->dims->num_dims * sizeof(unsigned int));
    unsigned int *dims_u = malloc(tensor->dims->num_dims * sizeof(unsigned int));
    for (size_t i = 0; i < tensor->dims->num_dims; i++) {
        perms_u[i] = (unsigned int) perms[i];
        dims_u[i] = (unsigned int) tensor->dims->dims[i];
    }
    unsigned int num_dims = tensor->dims->num_dims;
    cl_mem perms_u_cl = rml_cl_create_buffer(CL_MEM_READ_ONLY, tensor->dims->num_dims * sizeof(unsigned int));
    cl_mem dims_u_cl = rml_cl_create_buffer(CL_MEM_READ_ONLY, tensor->dims->num_dims * sizeof(unsigned int));
    rml_cl_enqueue_write_buffer(perms_u_cl, tensor->dims->num_dims * sizeof(unsigned int), perms_u);
    rml_cl_enqueue_write_buffer(dims_u_cl, tensor->dims->num_dims * sizeof(unsigned int), dims_u);

    rml_cl_set_kernel_arg(CL_OP_PERMUTE, rml_cl_typeof_tensor(tensor), 0, tensor->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_PERMUTE, rml_cl_typeof_tensor(tensor), 1, result->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_PERMUTE, rml_cl_typeof_tensor(tensor), 2, &perms_u_cl, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_PERMUTE, rml_cl_typeof_tensor(tensor), 3, &dims_u_cl, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_PERMUTE, rml_cl_typeof_tensor(tensor), 4, &num_dims, sizeof(unsigned int));
    rml_cl_enqueue_range_kernel(CL_OP_PERMUTE, rml_cl_typeof_tensor(tensor), &result->dims->flat_size);

    result->op_code = OP_CODE_PERMUTE;
    result->source_a = tensor;
    result->op_data = malloc(tensor->dims->num_dims * sizeof(size_t));
    for (size_t i = 0; i < tensor->dims->num_dims; i++) {
        *((size_t *) result->op_data) = perms[i];
    }

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

tensor_t *rml_cl_add_tensor(tensor_t *a, tensor_t *b) {
    tensor_t *a_orig = a, *b_orig = b;
    CAST_TENSORS_WIDEN(a, b);
    tensor_t *result = rml_cl_zeros_tensor(a->tensor_type, rml_clone_dims(a->dims));

    rml_cl_set_kernel_arg(CL_OP_ADD, rml_cl_typeof_tensor(a), 0, a->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_ADD, rml_cl_typeof_tensor(a), 1, b->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_ADD, rml_cl_typeof_tensor(a), 2, result->cl_mem, sizeof(cl_mem));
    rml_cl_enqueue_range_kernel(CL_OP_ADD, rml_cl_typeof_tensor(a), &result->dims->flat_size);

    result->op_code = OP_CODE_ADD;
    result->source_a = a_orig;
    result->source_b = b_orig;
    CLEANUP_CAST_TENSORS_WIDEN;

    return result;
}

tensor_t *rml_cl_sub_tensor(tensor_t *a, tensor_t *b) {
    tensor_t *a_orig = a, *b_orig = b;
    CAST_TENSORS_WIDEN(a, b);
    tensor_t *result = rml_cl_zeros_tensor(a->tensor_type, rml_clone_dims(a->dims));

    rml_cl_set_kernel_arg(CL_OP_SUB, rml_cl_typeof_tensor(a), 0, a->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_SUB, rml_cl_typeof_tensor(a), 1, b->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_SUB, rml_cl_typeof_tensor(a), 2, result->cl_mem, sizeof(cl_mem));
    rml_cl_enqueue_range_kernel(CL_OP_SUB, rml_cl_typeof_tensor(a), &result->dims->flat_size);

    result->op_code = OP_CODE_SUB;
    result->source_a = a_orig;
    result->source_b = b_orig;
    CLEANUP_CAST_TENSORS_WIDEN;

    return result;
}

tensor_t *rml_cl_mul_tensor(tensor_t *a, tensor_t *b) {
    tensor_t *a_orig = a, *b_orig = b;
    CAST_TENSORS_WIDEN(a, b);
    tensor_t *result = rml_cl_zeros_tensor(a->tensor_type, rml_clone_dims(a->dims));

    rml_cl_set_kernel_arg(CL_OP_MUL, rml_cl_typeof_tensor(a), 0, a->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_MUL, rml_cl_typeof_tensor(a), 1, b->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_MUL, rml_cl_typeof_tensor(a), 2, result->cl_mem, sizeof(cl_mem));
    rml_cl_enqueue_range_kernel(CL_OP_MUL, rml_cl_typeof_tensor(a), &result->dims->flat_size);

    result->op_code = OP_CODE_MUL;
    result->source_a = a_orig;
    result->source_b = b_orig;
    CLEANUP_CAST_TENSORS_WIDEN;

    return result;
}

tensor_t *rml_cl_div_tensor(tensor_t *a, tensor_t *b) {
    tensor_t *a_orig = a, *b_orig = b;
    CAST_TENSORS_WIDEN(a, b);
    tensor_t *result = rml_cl_zeros_tensor(a->tensor_type, rml_clone_dims(a->dims));

    rml_cl_set_kernel_arg(CL_OP_DIV, rml_cl_typeof_tensor(a), 0, a->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_DIV, rml_cl_typeof_tensor(a), 1, b->cl_mem, sizeof(cl_mem));
    rml_cl_set_kernel_arg(CL_OP_DIV, rml_cl_typeof_tensor(a), 2, result->cl_mem, sizeof(cl_mem));
    rml_cl_enqueue_range_kernel(CL_OP_DIV, rml_cl_typeof_tensor(a), &result->dims->flat_size);

    result->op_code = OP_CODE_DIV;
    result->source_a = a_orig;
    result->source_b = b_orig;
    CLEANUP_CAST_TENSORS_WIDEN;

    return result;
}

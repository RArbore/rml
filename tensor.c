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

#include "tensor.h"

#define RAND_GRANULARITY 1000000

dims_t *rml_create_dims(size_t count, ...) {
    dims_t *dims = malloc(sizeof(dims_t));
    dims->num_dims = count;
    dims->dims = malloc(count * sizeof(size_t));

    va_list ap;
    va_start(ap, count);
    for (size_t i = 0; i < count; i++) {
        dims->dims[i] = va_arg(ap, size_t);
    }
    va_end(ap);

    dims->flat_size = 1;
    for (size_t i = 0; i < dims->num_dims; i++) {
        assert(SIZE_MAX / dims->dims[i] >= dims->flat_size); // Check for overflow
        dims->flat_size *= dims->dims[i];
    }

    return dims;
}

dims_t *rml_clone_dims(dims_t *dims) {
    dims_t *clone = malloc(sizeof(dims_t));
    clone->num_dims = dims->num_dims;
    clone->dims = malloc(clone->num_dims * sizeof(size_t));
    for (size_t i = 0; i < clone->num_dims; i++) {
        clone->dims[i] = dims->dims[i];
    }

    clone->flat_size = 1;
    for (size_t i = 0; i < clone->num_dims; i++) {
        assert(SIZE_MAX / clone->dims[i] >= dims->flat_size); // Check for overflow
        clone->flat_size *= dims->dims[i];
    }

    return clone;
}

int rml_dims_equiv(dims_t *a, dims_t *b) {
    if (a->num_dims != b->num_dims || a->flat_size != b->flat_size) return 0;
    for (size_t i = 0; i < a->num_dims; i++) {
        if (a->dims[i] != b->dims[i]) return 0;
    }
    return 1;
}

void rml_free_dims(dims_t *dims) {
    free(dims->dims);
    free(dims);
}

void rml_print_dims(dims_t *dims) {
    for (size_t i = 0; i < dims->num_dims; i++) {
        printf("%lu", dims->dims[i]);
        if (i + 1 < dims->num_dims) printf(" ");
    }
    printf("\n");
}

tensor_t *rml_init_tensor(tensor_type_t type, dims_t *dims, void *data){
    tensor_t *tensor = malloc(sizeof(tensor_t));

    tensor->tensor_type = type;
    tensor->dims = dims;
    tensor->data = malloc(dims->flat_size * rml_sizeof_type(type));
    if (data != NULL) {
        for (size_t i = 0; i < dims->flat_size; i++) {
            SWITCH_ENUM_TYPES(type, COPY_VOID_POINTER, tensor->data, data, i, i);
        }
    }
    tensor->op_code = OP_CODE_CREATE;
    tensor->source_a = NULL;
    tensor->source_b = NULL;
    tensor->op_data = NULL;

    return tensor;
}

tensor_t *rml_create_tensor(tensor_type_t type, dims_t *dims, size_t count, ...){
    tensor_t *tensor = malloc(sizeof(tensor_t));

    tensor->tensor_type = type;
    tensor->dims = dims;
    tensor->data = malloc(dims->flat_size * rml_sizeof_type(type));
    assert(count == dims->flat_size);

    va_list ap;
    va_start(ap, count);
    for (size_t i = 0; i < count; i++) {
        SWITCH_ENUM_TYPES_VA(type, ap, tensor->data, i);
    }
    va_end(ap);
    tensor->op_code = OP_CODE_CREATE;
    tensor->source_a = NULL;
    tensor->source_b = NULL;
    tensor->op_data = NULL;

    return tensor;
}

tensor_t *rml_zeros_tensor(tensor_type_t type, dims_t *dims){
    tensor_t *tensor = malloc(sizeof(tensor_t));

    tensor->tensor_type = type;
    tensor->dims = dims;
    tensor->data = calloc(dims->flat_size, rml_sizeof_type(type));
    tensor->op_code = OP_CODE_CREATE;
    tensor->source_a = NULL;
    tensor->source_b = NULL;
    tensor->op_data = NULL;

    return tensor;
}

tensor_t *rml_ones_tensor(tensor_type_t type, dims_t *dims){
    tensor_t *tensor = malloc(sizeof(tensor_t));

    tensor->tensor_type = type;
    tensor->dims = dims;
    tensor->data = malloc(dims->flat_size * rml_sizeof_type(type));
    for (size_t i = 0; i < dims->flat_size; i++) {
        SWITCH_ENUM_TYPES(type, ASSIGN_VOID_POINTER, tensor->data, 1, i);
    }
    tensor->op_code = OP_CODE_CREATE;
    tensor->source_a = NULL;
    tensor->source_b = NULL;
    tensor->op_data = NULL;

    return tensor;
}

tensor_t *rml_rand_tensor(tensor_type_t type, dims_t *dims) {
    assert(type == TENSOR_TYPE_FLOAT || type == TENSOR_TYPE_DOUBLE || type == TENSOR_TYPE_LDOUBLE);
    tensor_t *tensor = malloc(sizeof(tensor_t));

    tensor->tensor_type = type;
    tensor->dims = dims;
    tensor->data = malloc(dims->flat_size * rml_sizeof_type(type));
    for (size_t i = 0; i < dims->flat_size; i++) {
        SWITCH_ENUM_TYPES(type, ASSIGN_VOID_POINTER, tensor->data, (long double) (rand() % RAND_GRANULARITY) / RAND_GRANULARITY, i);
    }
    tensor->op_code = OP_CODE_CREATE;
    tensor->source_a = NULL;
    tensor->source_b = NULL;
    tensor->op_data = NULL;

    return tensor;
}

tensor_t *rml_clone_tensor(tensor_t *tensor){
    if (tensor->tensor_type == TENSOR_TYPE_FLOAT || tensor->tensor_type == TENSOR_TYPE_DOUBLE) return rml_blas_clone_tensor(tensor);
    tensor_t *clone = rml_init_tensor(tensor->tensor_type, rml_clone_dims(tensor->dims), NULL);
    clone->op_code = OP_CODE_CLONE;
    clone->source_a = tensor;
    for (size_t i = 0; i < tensor->dims->flat_size; i++) {
        SWITCH_ENUM_TYPES(clone->tensor_type, COPY_VOID_POINTER, clone->data, tensor->data, i, i);
    }
    return clone;
}

void rml_free_tensor(tensor_t *tensor) {
    if (tensor == NULL) return;
    rml_free_dims(tensor->dims);
    free(tensor->data);
    free(tensor->op_data);
    free(tensor->cl_mem);
    free(tensor);
}

void rml_print_tensor(tensor_t *tensor) {
    for (size_t i = 0; i < tensor->dims->flat_size; i++) {
        PRINT_VOID_POINTER(tensor->tensor_type, tensor->data, i);
        if (i + 1 < tensor->dims->flat_size) printf(" ");
    }
    printf("\n");
}

void *rml_primitive_access_tensor(tensor_t *tensor, size_t *pos){
    size_t index = 0;
    for (size_t i = 0; i < tensor->dims->num_dims; i++) {
        index = index * tensor->dims->dims[i] + pos[i];
    }
    void *ret;
    SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, ret, 1);
    SWITCH_ENUM_TYPES(tensor->tensor_type, COPY_VOID_POINTER, ret, tensor->data, 0, index);
    return ret;
}

tensor_t *rml_matmul_tensor(tensor_t *a, tensor_t *b){
    assert(a->dims->num_dims == 2 && b->dims->num_dims == 2);
    assert(a->dims->dims[1] == b->dims->dims[0]);
    tensor_t *a_orig = a, *b_orig = b;
    CAST_TENSORS_WIDEN(a, b);
    if (a->tensor_type == TENSOR_TYPE_FLOAT || a->tensor_type == TENSOR_TYPE_DOUBLE) {
        tensor_t *result = rml_blas_matmul_tensor(a, b);
        CLEANUP_CAST_TENSORS_WIDEN;
        return result;
    }

    tensor_t *b_clone = rml_transpose_tensor(b);
    tensor_t *result = rml_zeros_tensor(a->tensor_type, rml_create_dims(2, a->dims->dims[0], b->dims->dims[1]));
    SWITCH_ENUM_TYPES(result->tensor_type, FAST_MATRIX_MULTIPLY, a, b_clone, result);
    rml_free_tensor(b_clone);
    result->op_code = OP_CODE_MATMUL;
    result->source_a = a_orig;
    result->source_b = b_orig;
    CLEANUP_CAST_TENSORS_WIDEN;

    return result;
}

tensor_t *rml_concat_tensor(tensor_t *a, tensor_t *b, size_t dim) {
    assert(a->dims->num_dims == b->dims->num_dims);
    assert(dim < a->dims->num_dims);
    for (size_t i = 0; i < a->dims->num_dims; i++) {
        if (i != dim) assert(a->dims->dims[i] == b->dims->dims[i]);
    }
    tensor_t *a_orig = a, *b_orig = b;
    CAST_TENSORS_WIDEN(a, b);
    dims_t *dims = rml_clone_dims(a->dims);
    dims->flat_size /= dims->dims[dim];
    dims->dims[dim] = a->dims->dims[dim] + b->dims->dims[dim];
    dims->flat_size *= dims->dims[dim];
    tensor_t *result = rml_init_tensor(a->tensor_type, dims, NULL);
    size_t pos_workspace[result->dims->num_dims];
    for (size_t i = 0; i < result->dims->num_dims; i++) {
        pos_workspace[i] = 0;
    }
    size_t a_index = 0;
    size_t b_index = 0;
    for (size_t i = 0; i < result->dims->flat_size; i++) {
        if (pos_workspace[dim] < a->dims->dims[dim]) {
            SWITCH_ENUM_TYPES(result->tensor_type, COPY_VOID_POINTER, result->data, a->data, i, a_index++);
        }
        else {
            SWITCH_ENUM_TYPES(result->tensor_type, COPY_VOID_POINTER, result->data, b->data, i, b_index++);
        }
        pos_workspace[result->dims->num_dims - 1]++;
        for (size_t d = result->dims->num_dims - 1;
             d > 0 && pos_workspace[d] >= result->dims->dims[d];
             pos_workspace[d] = 0, pos_workspace[d - 1]++, d--);
    }
    result->op_code = OP_CODE_CONCAT;
    result->source_a = a_orig;
    result->source_b = b_orig;
    result->op_data = malloc(sizeof(size_t));
    *((size_t *) result->op_data) = dim;
    CLEANUP_CAST_TENSORS_WIDEN;

    return result;
}

tensor_t *rml_slice_tensor(tensor_t *tensor, size_t *lower_bound, size_t *upper_bound) {
    dims_t *dims = malloc(sizeof(dims_t));
    dims->num_dims = tensor->dims->num_dims;
    dims->dims = malloc(dims->num_dims * sizeof(size_t));
    dims->flat_size = 1;
    for (size_t i = 0; i < tensor->dims->num_dims; i++) {
        assert(lower_bound[i] < upper_bound[i]);
        dims->dims[i] = upper_bound[i] - lower_bound[i];
        dims->flat_size *= dims->dims[i];
    }
    tensor_t *result = rml_init_tensor(tensor->tensor_type, dims, NULL);
    size_t pos_workspace[result->dims->num_dims], i_divided;
    for (size_t i = 0; i < result->dims->flat_size; i++) {
        i_divided = i;
        int reached_zero = 0;
        for (size_t d = result->dims->num_dims - 1; !reached_zero; d--) {
            if (d < result->dims->num_dims - 1) {
                i_divided /= result->dims->dims[d + 1];
            }
            pos_workspace[d] = i_divided % result->dims->dims[d] + lower_bound[d];
            if (d == 0) reached_zero = 1;
        }
        size_t old_pos = 0;
        for (size_t d = 0; d < tensor->dims->num_dims; d++) {
            size_t prev_mult = 0;
            if (d > 0) prev_mult = tensor->dims->dims[d];
            old_pos = old_pos * prev_mult + pos_workspace[d];
        }
        SWITCH_ENUM_TYPES(tensor->tensor_type, COPY_VOID_POINTER, result->data, tensor->data, i, old_pos);
    }
    result->op_code = OP_CODE_SLICE;
    result->source_a = tensor;
    result->op_data = malloc(2 * tensor->dims->num_dims * sizeof(size_t));
    for (size_t i = 0; i < tensor->dims->num_dims; i++) {
        *((size_t *) result->op_data) = lower_bound[i];
        *((size_t *) result->op_data) = upper_bound[i + tensor->dims->num_dims];
    }

    return result;
}

tensor_t *rml_assign_slice_tensor(tensor_t *a, tensor_t *b, size_t *lower_bound) {
    assert(a->dims->num_dims == b->dims->num_dims);
    tensor_t *result = rml_clone_tensor(a);
    size_t pos_workspace[b->dims->num_dims], i_divided;
    for (size_t i = 0; i < b->dims->flat_size; i++) {
        i_divided = i;
        int reached_zero = 0;
        for (size_t d = b->dims->num_dims - 1; !reached_zero; d--) {
            if (d < b->dims->num_dims - 1) {
                i_divided /= b->dims->dims[d + 1];
            }
            pos_workspace[d] = i_divided % b->dims->dims[d] + lower_bound[d];
            if (d == 0) reached_zero = 1;
        }
        size_t new_pos = 0;
        for (size_t d = 0; d < result->dims->num_dims; d++) {
            size_t prev_mult = 0;
            if (d > 0) prev_mult = result->dims->dims[d];
            new_pos = new_pos * prev_mult + pos_workspace[d];
        }
        SWITCH_2_ENUM_TYPES(result->tensor_type, b->tensor_type, CAST_VOID_POINTER, result->data, b->data, new_pos, i);
    }
    result->op_code = OP_CODE_ASSIGN_SLICE;
    result->source_a = a;
    result->source_b = b;
    result->op_data = malloc(b->dims->num_dims * sizeof(size_t));
    for (size_t i = 0; i < b->dims->num_dims; i++) {
        *((size_t *) result->op_data) = lower_bound[i];
    }

    return result;
}

tensor_t *rml_transpose_tensor(tensor_t *tensor) {
    assert(tensor->dims->num_dims == 2);
    tensor_t *result = rml_init_tensor(tensor->tensor_type, rml_clone_dims(tensor->dims), NULL);

    for (size_t r = 0; r < tensor->dims->dims[0]; r++) {
        for (size_t c = 0; c < tensor->dims->dims[1]; c++) {
            SWITCH_ENUM_TYPES(tensor->tensor_type, COPY_VOID_POINTER, result->data, tensor->data, c * tensor->dims->dims[0] + r, r * tensor->dims->dims[1] + c);
        }
    }
    size_t swap = result->dims->dims[0];
    result->dims->dims[0] = result->dims->dims[1];
    result->dims->dims[1] = swap;
    result->op_code = OP_CODE_TRANSPOSE;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_permute_tensor(tensor_t *tensor, size_t *perms) {
    size_t *new_dims = malloc(tensor->dims->num_dims * sizeof(size_t));
    for (size_t i = 0; i < tensor->dims->num_dims; i++) {
        new_dims[i] = tensor->dims->dims[perms[i]];
    }
    dims_t *new_dims_struct = rml_clone_dims(tensor->dims);
    free(new_dims_struct->dims);
    new_dims_struct->dims = new_dims;
    tensor_t *result = rml_init_tensor(tensor->tensor_type, new_dims_struct, NULL);
    size_t pos_workspace[tensor->dims->num_dims], i_divided;
    for (size_t i = 0; i < tensor->dims->flat_size; i++) {
        i_divided = i;
        int reached_zero = 0;
        for (size_t d = tensor->dims->num_dims - 1; !reached_zero; d--) {
            if (d < tensor->dims->num_dims - 1) {
                i_divided /= tensor->dims->dims[d + 1];
            }
            pos_workspace[d] = i_divided % tensor->dims->dims[d];
            if (d == 0) reached_zero = 1;
        }
        size_t new_pos = 0;
        for (size_t d = 0; d < tensor->dims->num_dims; d++) {
            size_t prev_mult = 0;
            if (d > 0) prev_mult = tensor->dims->dims[perms[d]];
            new_pos = new_pos * prev_mult + pos_workspace[perms[d]];
        }
        SWITCH_ENUM_TYPES(tensor->tensor_type, COPY_VOID_POINTER, result->data, tensor->data, new_pos, i);
    }
    result->op_code = OP_CODE_PERMUTE;
    result->source_a = tensor;
    result->op_data = malloc(tensor->dims->num_dims * sizeof(size_t));
    for (size_t i = 0; i < tensor->dims->num_dims; i++) {
        *((size_t *) result->op_data) = perms[i];
    }

    return result;
}

tensor_t *rml_reshape_tensor(tensor_t *tensor, size_t *new_dims, size_t count) {
    size_t flat_size_check = 1;
    for (size_t i = 0; i < count; i++) {
        flat_size_check *= new_dims[i];
    }
    assert(flat_size_check == tensor->dims->flat_size);
    tensor_t *result = rml_clone_tensor(tensor);
    dims_t *new_dims_struct = malloc(sizeof(dims_t));
    new_dims_struct->num_dims = count;
    new_dims_struct->flat_size = flat_size_check;
    new_dims_struct->dims = malloc(count * sizeof(size_t));
    for (size_t i = 0; i < count; i++) {
        new_dims_struct->dims[i] = new_dims[i];
    }
    rml_free_dims(result->dims);
    result->dims = new_dims_struct;
    result->op_code = OP_CODE_RESHAPE;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_cast_tensor(tensor_t *tensor, tensor_type_t type) {
    tensor_t *result = rml_init_tensor(type, rml_clone_dims(tensor->dims), NULL);
    result->op_code = OP_CODE_CAST;
    result->source_a = tensor;
    for (size_t i = 0; i < tensor->dims->flat_size; i++) {
        SWITCH_2_ENUM_TYPES(type, tensor->tensor_type, CAST_VOID_POINTER, result->data, tensor->data, i, i);
    }
    return result;
}

tensor_t *rml_add_tensor(tensor_t *a, tensor_t *b) {
    tensor_t *a_orig = a, *b_orig = b;
    CAST_TENSORS_WIDEN(a, b);
    if (a->tensor_type == TENSOR_TYPE_FLOAT || a->tensor_type == TENSOR_TYPE_DOUBLE) {
        tensor_t *result = rml_blas_add_tensor(a, b);
        CLEANUP_CAST_TENSORS_WIDEN;
        return result;
    }

    assert(rml_dims_equiv(a->dims, b->dims));
    tensor_t *result = rml_init_tensor(a->tensor_type, rml_clone_dims(a->dims), NULL);
    SWITCH_ENUM_TYPES(a->tensor_type, ADD_TENSORS, a, b, result);
    CLEANUP_CAST_TENSORS_WIDEN;
    result->op_code = OP_CODE_ADD;
    result->source_a = a_orig;
    result->source_b = b_orig;

    return result;
}

tensor_t *rml_sub_tensor(tensor_t *a, tensor_t *b) {
    tensor_t *a_orig = a, *b_orig = b;
    CAST_TENSORS_WIDEN(a, b);
    if (a->tensor_type == TENSOR_TYPE_FLOAT || a->tensor_type == TENSOR_TYPE_DOUBLE) {
        tensor_t *result = rml_blas_sub_tensor(a, b);
        CLEANUP_CAST_TENSORS_WIDEN;
        return result;
    }

    assert(rml_dims_equiv(a->dims, b->dims));
    tensor_t *result = rml_init_tensor(a->tensor_type, rml_clone_dims(a->dims), NULL);
    SWITCH_ENUM_TYPES(a->tensor_type, SUB_TENSORS, a, b, result);
    CLEANUP_CAST_TENSORS_WIDEN;
    result->op_code = OP_CODE_SUB;
    result->source_a = a_orig;
    result->source_b = b_orig;

    return result;
}

tensor_t *rml_mul_tensor(tensor_t *a, tensor_t *b) {
    tensor_t *a_orig = a, *b_orig = b;
    CAST_TENSORS_WIDEN(a, b);
    assert(rml_dims_equiv(a->dims, b->dims));
    tensor_t *result = rml_init_tensor(a->tensor_type, rml_clone_dims(a->dims), NULL);
    SWITCH_ENUM_TYPES(a->tensor_type, MUL_TENSORS, a, b, result);
    CLEANUP_CAST_TENSORS_WIDEN;
    result->op_code = OP_CODE_MUL;
    result->source_a = a_orig;
    result->source_b = b_orig;

    return result;
}

tensor_t *rml_div_tensor(tensor_t *a, tensor_t *b) {
    tensor_t *a_orig = a, *b_orig = b;
    CAST_TENSORS_WIDEN(a, b);
    assert(rml_dims_equiv(a->dims, b->dims));
    tensor_t *result = rml_init_tensor(a->tensor_type, rml_clone_dims(a->dims), NULL);
    SWITCH_ENUM_TYPES(a->tensor_type, DIV_TENSORS, a, b, result);
    CLEANUP_CAST_TENSORS_WIDEN;
    result->op_code = OP_CODE_DIV;
    result->source_a = a_orig;
    result->source_b = b_orig;

    return result;
}

tensor_t *rml_increment_tensor(tensor_t *a, void *scalar) {
    tensor_t *result = rml_clone_tensor(a);
    SWITCH_ENUM_TYPES(a->tensor_type, INCREMENT_TENSOR, a, scalar, result);
    result->op_code = OP_CODE_INCREMENT;
    result->source_a = a;
    result->op_data = malloc(rml_sizeof_type(a->tensor_type));
    SWITCH_ENUM_TYPES(a->tensor_type, COPY_VOID_POINTER, result->op_data, scalar, 0, 0);

    return result;
}

tensor_t *rml_scale_tensor(tensor_t *a, void *scalar) {
    if (a->tensor_type == TENSOR_TYPE_FLOAT || a->tensor_type == TENSOR_TYPE_DOUBLE) return rml_blas_scale_tensor(a, scalar);
    tensor_t *result = rml_clone_tensor(a);
    SWITCH_ENUM_TYPES(a->tensor_type, SCALE_TENSOR, a, scalar, result);
    result->op_code = OP_CODE_SCALE;
    result->source_a = a;
    result->op_data = malloc(rml_sizeof_type(a->tensor_type));
    SWITCH_ENUM_TYPES(a->tensor_type, COPY_VOID_POINTER, result->op_data, scalar, 0, 0);

    return result;
}

tensor_t *rml_floating_point_op_tensor(tensor_t *tensor, float (*f)(float), double (*d)(double), long double (*ld)(long double)) {
    assert(tensor->tensor_type == TENSOR_TYPE_FLOAT ||
           tensor->tensor_type == TENSOR_TYPE_DOUBLE ||
           tensor->tensor_type == TENSOR_TYPE_LDOUBLE);
    tensor_t *result = rml_init_tensor(tensor->tensor_type, rml_clone_dims(tensor->dims), NULL);
    switch (tensor->tensor_type) {
        case TENSOR_TYPE_FLOAT:
            for (size_t i = 0; i < tensor->dims->flat_size; i++) {
                ((float *) result->data)[i] = f(((float *) tensor->data)[i]);
            }
            break;
        case TENSOR_TYPE_DOUBLE:
            for (size_t i = 0; i < tensor->dims->flat_size; i++) {
                ((double *) result->data)[i] = d(((double *) tensor->data)[i]);
            }
            break;
        case TENSOR_TYPE_LDOUBLE:
            for (size_t i = 0; i < tensor->dims->flat_size; i++) {
                ((long double *) result->data)[i] = ld(((long double *) tensor->data)[i]);
            }
            break;
        default:
            break;
    }

    return result;
}

tensor_t *rml_exp_tensor(tensor_t *tensor) {
    tensor_t *result = rml_floating_point_op_tensor(tensor, expf, exp, expl);
    result->source_a = tensor;
    result->op_code = OP_CODE_EXP;

    return result;
}

tensor_t *rml_log_tensor(tensor_t *tensor) {
    tensor_t *result = rml_floating_point_op_tensor(tensor, logf, log, logl);
    result->source_a = tensor;
    result->op_code = OP_CODE_LOG;

    return result;
}

tensor_t *rml_pow_tensor(tensor_t *tensor, void *scalar) {
    assert(tensor->tensor_type == TENSOR_TYPE_FLOAT ||
           tensor->tensor_type == TENSOR_TYPE_DOUBLE ||
           tensor->tensor_type == TENSOR_TYPE_LDOUBLE);
    tensor_t *result = rml_init_tensor(tensor->tensor_type, rml_clone_dims(tensor->dims), NULL);
    switch (tensor->tensor_type) {
        case TENSOR_TYPE_FLOAT:
            for (size_t i = 0; i < tensor->dims->flat_size; i++) {
                ((float *) result->data)[i] = powf(((float *) tensor->data)[i], *((float *) scalar));
            }
            break;
        case TENSOR_TYPE_DOUBLE:
            for (size_t i = 0; i < tensor->dims->flat_size; i++) {
                ((double *) result->data)[i] = pow(((double *) tensor->data)[i], *((double *) scalar));
            }
            break;
        case TENSOR_TYPE_LDOUBLE:
            for (size_t i = 0; i < tensor->dims->flat_size; i++) {
                ((long double *) result->data)[i] = powl(((long double *) tensor->data)[i], *((long double *) scalar));
            }
            break;
        default:
            break;
    }
    result->op_code = OP_CODE_POW;
    result->source_a = tensor;
    result->op_data = malloc(rml_sizeof_type(tensor->tensor_type));
    SWITCH_ENUM_TYPES(tensor->tensor_type, COPY_VOID_POINTER, result->op_data, scalar, 0, 0);

    return result;
}

tensor_t *rml_sin_tensor(tensor_t *tensor) {
    tensor_t *result = rml_floating_point_op_tensor(tensor, sinf, sin, sinl);
    result->op_code = OP_CODE_SIN;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_cos_tensor(tensor_t *tensor) {
    tensor_t *result = rml_floating_point_op_tensor(tensor, cosf, cos, cosl);
    result->op_code = OP_CODE_COS;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_tan_tensor(tensor_t *tensor) {
    tensor_t *result = rml_floating_point_op_tensor(tensor, tanf, tan, tanl);
    result->op_code = OP_CODE_TAN;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_sinh_tensor(tensor_t *tensor) {
    tensor_t *result = rml_floating_point_op_tensor(tensor, sinhf, sinh, sinhl);
    result->op_code = OP_CODE_SINH;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_cosh_tensor(tensor_t *tensor) {
    tensor_t *result = rml_floating_point_op_tensor(tensor, coshf, cosh, coshl);
    result->op_code = OP_CODE_COSH;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_tanh_tensor(tensor_t *tensor) {
    tensor_t *result = rml_floating_point_op_tensor(tensor, tanhf, tanh, tanhl);
    result->op_code = OP_CODE_TANH;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_asin_tensor(tensor_t *tensor) {
    tensor_t *result = rml_floating_point_op_tensor(tensor, asinf, asin, asinl);
    result->op_code = OP_CODE_ASIN;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_acos_tensor(tensor_t *tensor) {
    tensor_t *result = rml_floating_point_op_tensor(tensor, acosf, acos, acosl);
    result->op_code = OP_CODE_ACOS;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_atan_tensor(tensor_t *tensor) {
    tensor_t *result = rml_floating_point_op_tensor(tensor, atanf, atan, atanl);
    result->op_code = OP_CODE_ATAN;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_asinh_tensor(tensor_t *tensor) {
    tensor_t *result = rml_floating_point_op_tensor(tensor, asinhf, asinh, asinhl);
    result->op_code = OP_CODE_ASINH;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_acosh_tensor(tensor_t *tensor) {
    tensor_t *result = rml_floating_point_op_tensor(tensor, acoshf, acosh, acoshl);
    result->op_code = OP_CODE_ACOSH;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_atanh_tensor(tensor_t *tensor) {
    tensor_t *result = rml_floating_point_op_tensor(tensor, atanhf, atanh, atanhl);
    result->op_code = OP_CODE_ATANH;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_abs_tensor(tensor_t *tensor) {
    tensor_t *result = rml_init_tensor(tensor->tensor_type, rml_clone_dims(tensor->dims), NULL);
    for (size_t i = 0; i< result->dims->flat_size; i++) {
        SWITCH_ENUM_TYPES(result->tensor_type, ABS_VOID_POINTER, result->data, tensor->data, i, i);
    }
    result->op_code = OP_CODE_ABS;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_clamp_tensor(tensor_t *tensor, void *min, void *max) {
    tensor_t *result = rml_clone_tensor(tensor);
    SWITCH_ENUM_TYPES(tensor->tensor_type, CLAMP_TENSOR, tensor, min, max, result);
    result->op_code = OP_CODE_CLAMP;
    result->source_a = tensor;
    result->op_data = malloc(sizeof(char) + 2 * rml_sizeof_type(tensor->tensor_type));
    *((char *) result->op_data) = 0;
    if (min != NULL) {
        SWITCH_ENUM_TYPES(tensor->tensor_type, COPY_VOID_POINTER, (void *) (((char *) result->op_data) + 1), min, 0, 0);
        *((char *) result->op_data) += 1;
    }
    if (max != NULL) {
        SWITCH_ENUM_TYPES(tensor->tensor_type, COPY_VOID_POINTER, (void *) (((char *) result->op_data) + 1), max, 1, 0);
        *((char *) result->op_data) += 2;
    }

    return result;
}

void *rml_max_tensor(tensor_t *tensor) {
    size_t max_pos = 0;
    for (size_t i = 0; i < tensor->dims->flat_size; i++) {
        int comp;
        SWITCH_ENUM_TYPES(tensor->tensor_type, COMPARE_VOID_POINTER, tensor->data, tensor->data, max_pos, i, comp);
        if (comp == -1) max_pos = i;
    }
    void *result;
    SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, result, 1);
    SWITCH_ENUM_TYPES(tensor->tensor_type, COPY_VOID_POINTER, result, tensor->data, 0, max_pos);

    return result;
}

void *rml_min_tensor(tensor_t *tensor) {
    size_t min_pos = 0;
    for (size_t i = 0; i < tensor->dims->flat_size; i++) {
        int comp;
        SWITCH_ENUM_TYPES(tensor->tensor_type, COMPARE_VOID_POINTER, tensor->data, tensor->data, min_pos, i, comp);
        if (comp == 1) min_pos = i;
    }
    void *result;
    SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, result, 1);
    SWITCH_ENUM_TYPES(tensor->tensor_type, COPY_VOID_POINTER, result, tensor->data, 0, min_pos);

    return result;
}

tensor_t *rml_sum_tensor(tensor_t *tensor) {
    tensor_t *result = rml_zeros_tensor(tensor->tensor_type, rml_create_dims(1, 1));
    for (size_t i = 0; i < tensor->dims->flat_size; i++) {
        SWITCH_ENUM_TYPES(tensor->tensor_type, INCREMENT_VOID_POINTER_PTR, result->data, 0, tensor->data, i);
    }
    result->op_code = OP_CODE_SUM;
    result->source_a = tensor;

    return result;
}

tensor_t *rml_one_hot_tensor(tensor_t *tensor, void *range) {
    assert(tensor->tensor_type != TENSOR_TYPE_FLOAT && tensor->tensor_type != TENSOR_TYPE_DOUBLE && tensor->tensor_type != TENSOR_TYPE_LDOUBLE);
    size_t range_s, index;
    SWITCH_ENUM_TYPES(tensor->tensor_type, CAST_VOID_POINTER_SWAPPED_TYPES, size_t, &range_s, range, 0, 0);

    tensor_t *result = rml_zeros_tensor(tensor->tensor_type, rml_create_dims(2, tensor->dims->flat_size, range_s));
    size_t one = 1;

    for (size_t i = 0; i < tensor->dims->flat_size; i++) {
        SWITCH_ENUM_TYPES(tensor->tensor_type, CAST_VOID_POINTER_SWAPPED_TYPES, size_t, &index, tensor->data, 0, i);
        SWITCH_ENUM_TYPES(tensor->tensor_type, CAST_VOID_POINTER, size_t, result->data, &one, i * range_s + index, 0);
    }
    result->op_code = OP_CODE_ONE_HOT;
    result->source_a = tensor;

    return result;
}

size_t rml_sizeof_type(tensor_type_t tensor_type) {
    switch (tensor_type) {
        case TENSOR_TYPE_BYTE:
            return sizeof(char);
        case TENSOR_TYPE_UBYTE:
            return sizeof(unsigned char);
        case TENSOR_TYPE_SHORT:
            return sizeof(short);
        case TENSOR_TYPE_USHORT:
            return sizeof(unsigned short);
        case TENSOR_TYPE_INT:
            return sizeof(int);
        case TENSOR_TYPE_UINT:
            return sizeof(unsigned int);
        case TENSOR_TYPE_LONG:
            return sizeof(long);
        case TENSOR_TYPE_ULONG:
            return sizeof(unsigned long);
        case TENSOR_TYPE_FLOAT:
            return sizeof(float);
        case TENSOR_TYPE_DOUBLE:
            return sizeof(double);
        case TENSOR_TYPE_LDOUBLE:
            return sizeof(long double);
    }
}

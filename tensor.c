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

tensor_t *rml_init_tensor(tensor_type_t type, dims_t *dims){
    tensor_t *tensor = malloc(sizeof(tensor_t));

    tensor->tensor_type = type;
    // TODO implement setting grad_graph and tensor_id
    tensor->dims = dims;
    tensor->data = malloc(dims->flat_size * rml_sizeof_type(type));

    return tensor;
}

tensor_t *rml_create_tensor(tensor_type_t type, dims_t *dims, size_t count, ...){
    tensor_t *tensor = malloc(sizeof(tensor_t));

    tensor->tensor_type = type;
    // TODO implement setting grad_graph and tensor_id
    tensor->dims = dims;
    tensor->data = malloc(dims->flat_size * rml_sizeof_type(type));
    assert(count == dims->flat_size);

    va_list ap;
    va_start(ap, count);
    for (size_t i = 0; i < count; i++) {
        SWITCH_ENUM_TYPES_VA(type, ap, tensor->data, i);
    }
    va_end(ap);

    return tensor;
}

tensor_t *rml_zeros_tensor(tensor_type_t type, dims_t *dims){
    tensor_t *tensor = malloc(sizeof(tensor_t));

    tensor->tensor_type = type;
    // TODO implement setting grad_graph and tensor_id
    tensor->dims = dims;
    tensor->data = calloc(dims->flat_size, rml_sizeof_type(type));

    return tensor;
}

tensor_t *rml_ones_tensor(tensor_type_t type, dims_t *dims){
    tensor_t *tensor = malloc(sizeof(tensor_t));

    tensor->tensor_type = type;
    // TODO implement setting grad_graph and tensor_id
    tensor->dims = dims;
    tensor->data = malloc(dims->flat_size * rml_sizeof_type(type));
    for (size_t i = 0; i < dims->flat_size; i++) {
        SWITCH_ENUM_TYPES(type, ASSIGN_VOID_POINTER, tensor->data, 1, i);
    }

    return tensor;
}

tensor_t *rml_rand_tensor(tensor_type_t type, dims_t *dims) {
    assert(type == TENSOR_TYPE_FLOAT || type == TENSOR_TYPE_DOUBLE || type == TENSOR_TYPE_LDOUBLE);
    tensor_t *tensor = malloc(sizeof(tensor_t));

    tensor->tensor_type = type;
    // TODO implement setting grad_graph and tensor_id
    tensor->dims = dims;
    tensor->data = malloc(dims->flat_size * rml_sizeof_type(type));
    for (size_t i = 0; i < dims->flat_size; i++) {
        SWITCH_ENUM_TYPES(type, ASSIGN_VOID_POINTER, tensor->data, (long double) (rand() % RAND_GRANULARITY) / RAND_GRANULARITY, i);
    }

    return tensor;
}

tensor_t *rml_clone_tensor(tensor_t *tensor){
    tensor_t *clone = rml_init_tensor(tensor->tensor_type, rml_clone_dims(tensor->dims));
    for (size_t i = 0; i < tensor->dims->flat_size; i++) {
        SWITCH_ENUM_TYPES(clone->tensor_type, COPY_VOID_POINTER, clone->data, tensor->data, i, i);
    }
    return clone;
}

void rml_free_tensor(tensor_t *tensor) {
    rml_free_dims(tensor->dims);
    free(tensor->data);
    free(tensor);
}

void rml_print_tensor(tensor_t *tensor) {
    for (size_t i = 0; i < tensor->dims->flat_size; i++) {
        PRINT_VOID_POINTER(tensor->tensor_type, tensor->data, i);
        if (i + 1 < tensor->dims->flat_size) printf(" ");
    }
    printf("\n");
}

void *rml_tensor_primitive_access(tensor_t *tensor, dims_t *dims){
    assert(dims->num_dims == tensor->dims->num_dims);
    size_t index = 0;
    for (size_t i = 0; i < dims->num_dims; i++) {
        index = index * tensor->dims->dims[i] + dims->dims[i];
    }
    return INDEX_VOID_POINTER(tensor->tensor_type, tensor->data, index);
}

tensor_t *rml_tensor_matmul_naive(tensor_t *a, tensor_t *b){
    assert(a->dims->num_dims == 2 && b->dims->num_dims == 2);
    assert(a->dims->dims[1] == b->dims->dims[0]);
    CAST_TENSORS_WIDEN(a, b)

    tensor_t *result = rml_zeros_tensor(a->tensor_type, rml_create_dims(2, a->dims->dims[0], b->dims->dims[1]));
    for (size_t r = 0; r < result->dims->dims[0]; r++) {
        for (size_t c = 0; c < result->dims->dims[1]; c++) {
            size_t index_res = r * result->dims->dims[1] + c;
            for (size_t i = 0; i < a->dims->dims[1]; i++) {
                size_t index_a = r * a->dims->dims[1] + i;
                size_t index_b = i * b->dims->dims[1] + c;
                SWITCH_ENUM_TYPES(result->tensor_type, ACCUM_VOID_POINTERS, a->data, b->data, result->data, index_a, index_b, index_res);
            }
        }
    }

    return result;
}

tensor_t *rml_tensor_matmul(tensor_t *a, tensor_t *b){
    assert(a->dims->num_dims == 2 && b->dims->num_dims == 2);
    assert(a->dims->dims[1] == b->dims->dims[0]);
    CAST_TENSORS_WIDEN(a, b)

    tensor_t *b_clone = rml_clone_tensor(b);
    rml_tensor_transpose_inplace(b_clone);
    tensor_t *result = rml_zeros_tensor(a->tensor_type, rml_create_dims(2, a->dims->dims[0], b->dims->dims[1]));
    SWITCH_ENUM_TYPES(result->tensor_type, FAST_MATRIX_MULTIPLY, a, b_clone, result);
    free(b_clone);

    return result;
}

tensor_t *rml_tensor_transpose_inplace(tensor_t *tensor) {
    assert(tensor->dims->num_dims == 2);
    void *new;
    SWITCH_ENUM_TYPES(tensor->tensor_type, MALLOC_VOID_POINTER, new, tensor->dims->flat_size);

    for (size_t r = 0; r < tensor->dims->dims[0]; r++) {
        for (size_t c = 0; c < tensor->dims->dims[1]; c++) {
            SWITCH_ENUM_TYPES(tensor->tensor_type, COPY_VOID_POINTER, new, tensor->data, c * tensor->dims->dims[0] + r, r * tensor->dims->dims[1] + c);
        }
    }
    free(tensor->data);
    tensor->data = new;
    size_t swap = tensor->dims->dims[0];
    tensor->dims->dims[0] = tensor->dims->dims[1];
    tensor->dims->dims[1] = swap;

    return tensor;
}

tensor_t *rml_tensor_permute_inplace(tensor_t *tensor) {

}

tensor_t *rml_cast_tensor_inplace(tensor_t *tensor, tensor_type_t type){
    if (tensor->tensor_type == type) return tensor;
    void *new;
    SWITCH_ENUM_TYPES(type, MALLOC_VOID_POINTER, new, tensor->dims->flat_size);
    for (size_t i = 0; i < tensor->dims->flat_size; i++) {
        SWITCH_2_ENUM_TYPES(type, tensor->tensor_type, CAST_VOID_POINTER, new, tensor->data, i, i);
    }
    free(tensor->data);
    tensor->data = new;
    tensor->tensor_type = type;
    return tensor;
}

tensor_t *rml_tensor_add_inplace(tensor_t *a, tensor_t *b){
    CAST_TENSORS_WIDEN(a, b)
    assert(rml_dims_equiv(a->dims, b->dims));
    SWITCH_ENUM_TYPES(a->tensor_type, ADD_TENSORS, a, b, a);
    return a;
}

tensor_t *rml_tensor_mul_inplace(tensor_t *a, tensor_t *b){
    CAST_TENSORS_WIDEN(a, b)
    assert(rml_dims_equiv(a->dims, b->dims));
    SWITCH_ENUM_TYPES(a->tensor_type, MUL_TENSORS, a, b, a);
    return a;
}

tensor_t *rml_concat_inplace(tensor_t *a, tensor_t *b, unsigned char dim){

}

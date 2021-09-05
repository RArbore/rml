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

#include "internal.h"
#include "tensor.h"
#include "rml.h"

dims_t *rml_create_dims(int count, ...) {
    dims_t *dims = malloc(sizeof(dims_t));
    dims->num_dims = count;
    dims->dims = malloc(count * sizeof(size_t));

    va_list ap;
    va_start(ap, count);
    for (int i = 0; i < count; i++) {
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
    for (size_t i = 0; i < a->flat_size; i++) {
        if (a->dims[i] != b->dims[i]) return 0;
    }
    return 1;
}

void rml_free_dims(dims_t *dims) {
    free(dims->dims);
    free(dims);
}

tensor_t *rml_create_tensor(tensor_type_t type, dims_t *dims){
    tensor_t *tensor = malloc(sizeof(tensor_t));

    tensor->tensor_type = type;
    // TODO implement setting grad_graph and tensor_id
    tensor->dims = dims;
    tensor->data = malloc(dims->flat_size * rml_sizeof_type(type));

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

tensor_t *rml_clone_tensor(tensor_t *tensor){
    tensor_t *clone = rml_create_tensor(tensor->tensor_type, rml_clone_dims(tensor->dims));
    for (size_t i = 0; i < tensor->dims->flat_size; i++) {
        SWITCH_ENUM_TYPES(clone->tensor_type, COPY_VOID_POINTER, clone->data, tensor->data, i);
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

tensor_t *rml_tensor_matmul(tensor_t *a, tensor_t *b){

}

tensor_t *rml_cast_tensor_inplace(tensor_t *tensor, tensor_type_t type){
    if (tensor->tensor_type == type) return tensor;
    void *new;
    SWITCH_ENUM_TYPES(type, MALLOC_VOID_POINTER, new, tensor->dims->flat_size);
    for (size_t i = 0; i < tensor->dims->flat_size; i++) {
        SWITCH_2_ENUM_TYPES(type, tensor->tensor_type, CAST_VOID_POINTER, new, tensor->data, i);
    }
    free(tensor->data);
    tensor->data = new;
    return tensor;
}

tensor_t *rml_tensor_add_inplace(tensor_t *a, tensor_t *b){
    if (a->tensor_type > b->tensor_type) {
        rml_cast_tensor_inplace(b, a->tensor_type);
    }
    else if (a->tensor_type < b->tensor_type) {
        rml_cast_tensor_inplace(a, b->tensor_type);
    }
    assert(a->tensor_type == b->tensor_type);
    assert(rml_dims_equiv(a->dims, b->dims));
    SWITCH_ENUM_TYPES(a->tensor_type, ADD_TENSORS, a, b, a);
    return a;
}

tensor_t *rml_tensor_mul_inplace(tensor_t *a, tensor_t *b){

}

tensor_t *rml_concat_inplace(tensor_t *a, tensor_t *b, unsigned char dim){

}

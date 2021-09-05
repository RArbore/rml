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

    return dims;
}

dims_t *rml_clone_dims(dims_t *dims) {
    dims_t *clone = malloc(sizeof(dims_t));
    clone->num_dims = dims->num_dims;
    clone->dims = malloc(clone->num_dims * sizeof(size_t));
    for (size_t i = 0; i < clone->num_dims; i++) {
        clone->dims[i] = dims->dims[i];
    }
    return clone;
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
    size_t elements = 1;
    for (size_t i = 0; i < dims->num_dims; i++) {
        elements *= dims->dims[i];
    }
    tensor->data = malloc(elements * rml_sizeof_type(type));

    return tensor;
}

tensor_t *rml_zeros_tensor(tensor_type_t type, dims_t *dims){
    tensor_t *tensor = malloc(sizeof(tensor_t));

    tensor->tensor_type = type;
    // TODO implement setting grad_graph and tensor_id
    tensor->dims = dims;
    size_t elements = 1;
    for (size_t i = 0; i < dims->num_dims; i++) {
        elements *= dims->dims[i];
    }
    tensor->data = calloc(elements, rml_sizeof_type(type));

    return tensor;
}

tensor_t *rml_ones_tensor(tensor_type_t type, dims_t *dims){
    tensor_t *tensor = malloc(sizeof(tensor_t));

    tensor->tensor_type = type;
    // TODO implement setting grad_graph and tensor_id
    tensor->dims = dims;
    size_t elements = 1;
    for (size_t i = 0; i < dims->num_dims; i++) {
        elements *= dims->dims[i];
    }
    tensor->data = malloc(elements * rml_sizeof_type(type));
    for (size_t i = 0; i < elements; i++) {
        void *data = tensor->data + i;
        SWITCH_ENUM_TYPES(type, ASSIGN_VOID_POINTER, data, 1);
    }

    return tensor;
}

tensor_t *rml_clone_tensor(tensor_t *tensor){

}

void rml_free_tensor(tensor_t *tensor) {
    rml_free_dims(tensor->dims);
    free(tensor->data);
    free(tensor);
}

void *rml_tensor_primitive_access(tensor_t *tensor, dims_t *dims){

}

tensor_t *rml_tensor_matmul(tensor_t *a, tensor_t *b){

}

tensor_t *rml_cast_tensor_inplace(tensor_t *tensor, tensor_type_t type){

}

tensor_t *rml_tensor_add_inplace(tensor_t *a, tensor_t *b){

}

tensor_t *rml_tensor_mul_inplace(tensor_t *a, tensor_t *b){

}

tensor_t *rml_concat_inplace(tensor_t *a, tensor_t *b, unsigned char dim){

}

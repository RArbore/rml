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

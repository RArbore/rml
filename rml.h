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

#ifndef RML_H_
#define RML_H_

#include <stdlib.h>
#include <stdarg.h>

// Definition of different types of tensors (tensors can have any of the C primitive types as elements, enum used to store internally which primitive type is currently used by tensor)
typedef enum {
    TENSOR_TYPE_BYTE = 0x00,
    TENSOR_TYPE_UBYTE,
    TENSOR_TYPE_SHORT,
    TENSOR_TYPE_USHORT,
    TENSOR_TYPE_INT,
    TENSOR_TYPE_UINT,
    TENSOR_TYPE_LONG,
    TENSOR_TYPE_ULONG,
    TENSOR_TYPE_FLOAT,
    TENSOR_TYPE_DOUBLE,
    TENSOR_TYPE_LDOUBLE,
    TENSOR_TYPE_COUNT
} tensor_type_t;

// Definition of different types for dimension storage (dimension sizes for tensors are a subset of C primitives, just unsigned fixed point primitives)
typedef enum {
    DIM_TYPE_UBYTE = 0x00,
    DIM_TYPE_USHORT,
    DIM_TYPE_UINT,
    DIM_TYPE_ULONG,
    DIM_TYPE_COUNT
} dim_type_t;

// Tensor struct - direct access to any of a tensor's elements should be avoided in favor of using library methods
typedef struct {
    tensor_type_t tensor_type;
    dim_type_t dim_type;
    unsigned char num_dims;
    unsigned char grad_graph;
    unsigned long tensor_id;
    void *dims;
    void *data;
} tensor_t;

// Create a tensor with undefined elements
extern tensor_t *rml_create_tensor(tensor_type_t type, int count, ...);

// Create a tensor with all 0 elements
extern tensor_t *rml_zeros_tensor(tensor_type_t type, int count, ...);

// Create a tensor with all 1 elements
extern tensor_t *rml_ones_tensor(tensor_type_t type, int count, ...);

// Clone a tensor
extern tensor_t *rml_clone_tensor(tensor_t *tensor);

// Access a single tensor element - this WILL break gradient graph, rml_tensor_access should be used to preserve gradient graph
extern void *rml_tensor_primitive_access(tensor_t *tensor, ...);

// Matrix multiply 2 tensors (asserted that both tensors are 2d and dimensions work for matrix multiplication)
extern tensor_t *rml_tensor_matmul(tensor_t *a, tensor_t *b);

// Cast a tensor to a different type (inplace, pointer to a returned only for convenience)
extern tensor_t *rml_cast_tensor_inplace(tensor_t *tensor, tensor_type_t type);

// Element-wise add tensor b to tensor a (inplace, pointer to a returned only for convenience)
extern tensor_t *rml_tensor_add_inplace(tensor_t *a, tensor_t *b);

// Element-wise multiply tensor a by tensor b (inplace, pointer to a returned only for convenience)
extern tensor_t *rml_tensor_mul_inplace(tensor_t *a, tensor_t *b);

// Concatenate tensor b to tensor a (inplace, pointer to a returned only for convenience)
extern tensor_t *rml_concat_inplace(tensor_t *a, tensor_t *b, unsigned char dim);

#endif // RML_H_

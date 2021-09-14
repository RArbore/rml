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

/////////////
// TENSORS //
/////////////

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
    TENSOR_TYPE_LDOUBLE
} tensor_type_t;

typedef struct {
    size_t num_dims;
    size_t *dims;
    size_t flat_size;
} dims_t;

// Tensor struct - direct access to any of a tensor's elements should be avoided in favor of using library methods
typedef struct {
    tensor_type_t tensor_type;
    size_t grad_graph;
    size_t tensor_id;
    dims_t *dims;
    void *data;
} tensor_t;

// Create a dimensions struct (variadic)
extern dims_t *rml_create_dims(size_t count, ...);

// Clone a dimensions struct
extern dims_t *rml_clone_dims(dims_t *dims);

// Check equivalence between dimensions structs
extern int rml_dims_equiv(dims_t *a, dims_t *b);

// Free a dimensions struct
extern void rml_free_dims(dims_t *dims);

// Print a dimensions struct to stdout
extern void rml_print_dims(dims_t *dims);

// Create a tensor with undefined elements
extern tensor_t *rml_init_tensor(tensor_type_t type, dims_t *dims, void *data);

// Create a tensor with specified elements (behavior undefined if number of arguments provided isn't equal to flat size of tensor)
extern tensor_t *rml_create_tensor(tensor_type_t type, dims_t *dims, size_t count, ...);

// Create a tensor with all 0 elements
extern tensor_t *rml_zeros_tensor(tensor_type_t type, dims_t *dims);

// Create a tensor with all 1 elements
extern tensor_t *rml_ones_tensor(tensor_type_t type, dims_t *dims);

// Create a tensor with all random elements, uniform [0-1)
extern tensor_t *rml_rand_tensor(tensor_type_t type, dims_t *dims);

// Clone a tensor
extern tensor_t *rml_clone_tensor(tensor_t *tensor);

// Clone a tensor
extern tensor_t *rml_blas_clone_tensor(tensor_t *tensor);

// Free a tensor
extern void rml_free_tensor(tensor_t *tensor);

// Print a tensor to stdout
extern void rml_print_tensor(tensor_t *tensor);

// Access a single tensor element - this WILL break gradient graph, rml_tensor_access should be used to preserve gradient graph
extern void *rml_primitive_access_tensor(tensor_t *tensor, size_t *pos);

// Matrix multiply 2 tensors (asserted that both tensors are 2d and dimensions work for matrix multiplication, O(n^3) implementation)
extern tensor_t *rml_matmul_naive_tensor(tensor_t *a, tensor_t *b);

// Matrix multiply 2 tensors (asserted that both tensors are 2d and dimensions work for matrix multiplication, O(n^3) implementation w/ some memory optimizations)
extern tensor_t *rml_matmul_tensor(tensor_t *a, tensor_t *b);

// Matrix multiply 2 tensors (asserted that both tensors are 2d and dimensions work for matrix multiplication, use BLAS)
extern tensor_t *rml_blas_matmul_tensor(tensor_t *a, tensor_t *b);

// Concatenate tensor b to tensor a
extern tensor_t *rml_concat_tensor(tensor_t *a, tensor_t *b, size_t dim);

// Slice a tensor (index section)
extern tensor_t *rml_slice_tensor(tensor_t *tensor, size_t *lower_bound, size_t *upper_bound);

// Assign a slice of tensor a the values in tensor b
extern tensor_t *rml_assign_slice_tensor(tensor_t *a, tensor_t *b, size_t *lower_bound);

// Transpose a matrix (asserted that tensor is 2d, for more general form, see rml_tensor_permute_inplace)
extern tensor_t *rml_transpose_tensor(tensor_t *tensor);

// Permute axes of a tensor
extern tensor_t *rml_permute_tensor(tensor_t *tensor, size_t *perms);

// Reshape tensor
extern tensor_t *rml_reshape_tensor(tensor_t *tensor, size_t *new_dims, size_t count);

// Cast a tensor to a different type
extern tensor_t *rml_cast_tensor(tensor_t *tensor, tensor_type_t type);

// Element-wise add tensor b to tensor a
extern tensor_t *rml_add_tensor(tensor_t *a, tensor_t *b);

// Element-wise add tensor b to tensor a
extern tensor_t *rml_blas_add_tensor(tensor_t *a, tensor_t *b);

// Element-wise subtract tensor b to tensor a
extern tensor_t *rml_sub_tensor(tensor_t *a, tensor_t *b);

// Element-wise subtract tensor b to tensor a
extern tensor_t *rml_blas_sub_tensor(tensor_t *a, tensor_t *b);

// Element-wise multiply tensor a by tensor b
extern tensor_t *rml_mul_tensor(tensor_t *a, tensor_t *b);

// Element-wise divide tensor a by tensor b
extern tensor_t *rml_div_tensor(tensor_t *a, tensor_t *b);

// Element-wise add to tensor by a scalar
extern tensor_t *rml_increment_tensor(tensor_t *a, void *scalar);

// Element-wise multiply tensor by a scalar
extern tensor_t *rml_scale_tensor(tensor_t *a, void *scalar);

// Element-wise multiply tensor by a scalar
extern tensor_t *rml_blas_scale_tensor(tensor_t *a, void *scalar);

// Element-wise exponentiation of tensor
extern tensor_t *rml_exp_tensor(tensor_t *tensor);

// Element-wise logarithm of tensor
extern tensor_t *rml_log_tensor(tensor_t *tensor);

// Element-wise power of tensor
extern tensor_t *rml_pow_tensor(tensor_t *tensor, void *scalar);

// Element-wise sin of tensor
extern tensor_t *rml_sin_tensor(tensor_t *tensor);

// Element-wise cos of tensor
extern tensor_t *rml_cos_tensor(tensor_t *tensor);

// Element-wise tan of tensor
extern tensor_t *rml_tan_tensor(tensor_t *tensor);

// Element-wise sinh of tensor
extern tensor_t *rml_sinh_tensor(tensor_t *tensor);

// Element-wise cosh of tensor
extern tensor_t *rml_cosh_tensor(tensor_t *tensor);

// Element-wise tanh of tensor
extern tensor_t *rml_tanh_tensor(tensor_t *tensor);

// Element-wise asin of tensor
extern tensor_t *rml_asin_tensor(tensor_t *tensor);

// Element-wise acos of tensor
extern tensor_t *rml_acos_tensor(tensor_t *tensor);

// Element-wise atan of tensor
extern tensor_t *rml_atan_tensor(tensor_t *tensor);

// Element-wise asinh of tensor
extern tensor_t *rml_asinh_tensor(tensor_t *tensor);

// Element-wise acosh of tensor
extern tensor_t *rml_acosh_tensor(tensor_t *tensor);

// Element-wise atanh of tensor
extern tensor_t *rml_atanh_tensor(tensor_t *tensor);

// Element-wise absolute value of tensor
extern tensor_t *rml_abs_tensor(tensor_t *tensor);

// Clamp values of a tensor
extern tensor_t *rml_clamp_tensor(tensor_t *tensor, void *min, void *max);

// Find maximum value of tensor
extern void *rml_max_tensor(tensor_t *tensor);

// Find minimum value of tensor
extern void *rml_min_tensor(tensor_t *tensor);

// Find sum of all values of tensor
extern void *rml_sum_tensor(tensor_t *tensor);

// Read tensor to a csv file without metadata
extern tensor_t *rml_read_tensor_csv_raw(char *filename, tensor_type_t tensor_type, dims_t *dims);

// Read tensor to a csv file with metadata
extern tensor_t *rml_read_tensor_csv_full(char *filename);

// Write tensor to a csv file without metadata
extern void rml_write_tensor_csv_raw(char *filename, tensor_t *tensor);

// Write tensor to a csv file with metadata
extern void rml_write_tensor_csv_full(char *filename, tensor_t *tensor);

// Softmax of tensor
extern tensor_t *rml_softmax_tensor(tensor_t *tensor);

// ReLU of tensor
extern tensor_t *rml_relu_tensor(tensor_t *tensor);

// LeakyReLU of tensor
extern tensor_t *rml_leakyrelu_tensor(tensor_t *tensor, void *mult);

// Cross entropy loss between prediction and label tensors
extern tensor_t *rml_cross_entropy_loss_tensor(tensor_t *pred, tensor_t *label);

#endif // RML_H_

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

#ifndef CL_HELPERS_H_
#define CL_HELPERS_H_

#define CL_TARGET_OPENCL_VERSION 300

#include <string.h>
#include <CL/cl.h>
#include <stdio.h>

#include "cl_kernels.h"
#include "tensor.h"
#include "rml.h"

typedef enum {
    CL_TYPE_FLOAT = 0x00,
    CL_TYPE_DOUBLE,
} cl_type_t;

typedef enum {
    CL_OP_MATMUL = 0x00,
    CL_OP_CONCAT,
    CL_OP_SLICE,
    CL_OP_ASSIGN_SLICE,
    CL_OP_TRANSPOSE,
    CL_OP_PERMUTE,
    CL_OP_CAST_FLOAT,
    CL_OP_CAST_DOUBLE,
    CL_OP_ADD,
    CL_OP_SUB,
    CL_OP_MUL,
    CL_OP_DIV,
    CL_OP_INCREMENT,
    CL_OP_SCALE,
    CL_OP_EXP,
    CL_OP_LOG,
    CL_OP_POW,
    CL_OP_SIN,
    CL_OP_COS,
    CL_OP_TAN,
    CL_OP_SINH,
    CL_OP_COSH,
    CL_OP_TANH,
    CL_OP_ASIN,
    CL_OP_ACOS,
    CL_OP_ATAN,
    CL_OP_ASINH,
    CL_OP_ACOSH,
    CL_OP_ATANH,
    CL_OP_ABS,
    CL_OP_CLAMP,
    CL_OP_SUM,
    CL_OP_MAX,
    CL_OP_MIN,
} cl_op_t;

cl_mem rml_cl_create_buffer(int mem_properties, size_t size);

void rml_cl_enqueue_read_buffer(cl_mem buffer, size_t size, void *data);

void rml_cl_enqueue_write_buffer(cl_mem buffer, size_t size, void *data);

void rml_cl_enqueue_clone_buffer(cl_mem buffer_src, cl_mem buffer_dest, size_t size);

void rml_cl_enqueue_fill_buffer(cl_mem buffer, void *pattern, size_t pattern_size, size_t size);

void rml_cl_set_kernel_arg(cl_op_t kernel, cl_type_t tensor_type, size_t arg_index, void *arg, size_t size_of_arg);

void rml_cl_enqueue_range_kernel(cl_op_t kernel, cl_type_t tensor_type, size_t *op_size);

void rml_cl_finish();

void rml_cl_free_buffer(cl_mem buffer);

int rml_cl_same_device(size_t num, ...);

cl_type_t rml_cl_typeof_tensor(tensor_t *tensor);

#endif // CL_HELPERS_H_

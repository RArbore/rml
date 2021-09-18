#ifndef CL_KERNELS_H_
#define CL_KERNELS_H_

#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>
#include <stdio.h>

#include "rml.h"

cl_mem rml_cl_create_buffer(int mem_properties, size_t size);

void rml_cl_enqueue_read_buffer(cl_mem buffer, size_t size, void *data);

void rml_cl_enqueue_write_buffer(cl_mem buffer, size_t size, void *data);

void rml_cl_set_kernel_arg(op_code_t op_code, tensor_type_t tensor_type, size_t arg_index, cl_mem *buffer);

void rml_cl_enqueue_range_kernel(op_code_t op_code, tensor_type_t tensor_type, size_t op_size);

void rml_cl_free_buffer(cl_mem buffer);

#endif // CL_KERNELS_H_

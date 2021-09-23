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

cl_mem rml_cl_create_buffer(int mem_properties, size_t size);

void rml_cl_enqueue_read_buffer(cl_mem buffer, size_t size, void *data);

void rml_cl_enqueue_write_buffer(cl_mem buffer, size_t size, void *data);

void rml_cl_enqueue_clone_buffer(cl_mem buffer_src, cl_mem buffer_dest, size_t size);

void rml_cl_set_kernel_arg(unsigned short kernel, unsigned short tensor_type, size_t arg_index, cl_mem *buffer);

void rml_cl_enqueue_range_kernel(unsigned short kernel, unsigned short tensor_type, size_t op_size);

void rml_cl_finish();

void rml_cl_free_buffer(cl_mem buffer);

#endif // CL_HELPERS_H_

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

#include "cl_helpers.h"

#define NUM_TYPES 2
#define NUM_OP_CODES 42

cl_context context;
cl_command_queue command_queue;
cl_kernel kernels[NUM_OP_CODES][NUM_TYPES];

const char *type_names[] = {
    "float",
    "double",
};

const char *cl_max_arr_size = "1024";

const char *kernel_names[] = {
    "",
    "",
    "rml_clone",
    "rml_matmul",
    "rml_concat",
    "rml_slice",
    "rml_assign_slice",
    "rml_transpose",
    "rml_permute",
    "",
    "",
    "rml_add",
    "rml_sub",
    "rml_mul",
    "rml_div",
    "rml_increment",
    "rml_scale",
    "rml_exp",
    "rml_log",
    "rml_pow",
    "rml_sin",
    "rml_cos",
    "rml_tan",
    "rml_sinh",
    "rml_cosh",
    "rml_tanh",
    "rml_asin",
    "rml_acos",
    "rml_atan",
    "rml_asinh",
    "rml_acosh",
    "rml_atanh",
    "rml_abs",
    "rml_clamp",
    "rml_sum",
    "rml_one_hot",
    "rml_max",
    "rml_min",
    "",
    "",
    "",
    "",
};

static void rml_cl_kernel_init(cl_device_id device_id) {
    char *program_processed[NUM_TYPES];
    int max_arr_size_len = strlen(cl_max_arr_size);
    for (size_t t = 0; t < NUM_TYPES; t++) {
        int type_len = strlen(type_names[t]);
        program_processed[t] = malloc(2 * strlen(rml_cl_program) * sizeof(char));
        char *pos_type = strstr(rml_cl_program, "TYPE");
        char *pos_arr_size = strstr(rml_cl_program, "MAX_ARR_SIZE");
        size_t read_index = 0, write_index = 0;
        char read = rml_cl_program[read_index];
        while (read != '\0') {
            if (&rml_cl_program[read_index] == pos_type) {
                for (int c = 0; c < type_len; c++) {
                    program_processed[t][write_index++] = type_names[t][c];
                }
                read_index += 4;
                pos_type = strstr(pos_type + 4, "TYPE");
            }
            else if (&rml_cl_program[read_index] == pos_arr_size) {
                for (int c = 0; c < max_arr_size_len; c++) {
                    program_processed[t][write_index++] = cl_max_arr_size[c];
                }
                read_index += 12;
                pos_arr_size = strstr(pos_arr_size + 12, "MAX_ARR_SIZE");
            }
            else {
                program_processed[t][write_index++] = read;
                read_index++;
            }
            read = rml_cl_program[read_index];
        }
        program_processed[t][write_index] = '\0';

        cl_int err;
        cl_program program = clCreateProgramWithSource(context, 1, (const char **) &program_processed[t], NULL, &err);
        if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
            printf("Error building %s cl program\n", type_names[t]);
            char buffer[65536];
            size_t length;
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
            printf("%s\n", buffer);
            return;
        }

        for (size_t op = 0; op < NUM_OP_CODES; op++) {
            if (kernel_names[op][0] != '\0') {
                kernels[op][t] = clCreateKernel(program, kernel_names[op], &err);
                if (err != CL_SUCCESS) {
                    printf("Couldn't create kernel %s\n", kernel_names[op]);
                }
            }
        }
    }
}

void rml_cl_init() {
    cl_uint num_of_platforms = 0;
    cl_platform_id platform_id;
    int cl_status = clGetPlatformIDs(1, &platform_id, &num_of_platforms);
    if (cl_status != CL_SUCCESS) {
        printf("Unable to get platform_id for OpenCL (OpenCL initialization).");
        return;
    }

    cl_uint num_of_devices = 0;
    cl_device_id device_id;
    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices) != CL_SUCCESS) {
        printf("Unable to get device_id for OpenCL (finding a gpu).");
        return;
    }

    cl_context_properties properties[3];
    properties[0]= CL_CONTEXT_PLATFORM;
    properties[1]= (cl_context_properties) platform_id;
    properties[2]= 0;

    cl_int err;
    context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);
    command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &err);

    rml_cl_kernel_init(device_id);
}

void rml_cpu_to_cl_tensor(tensor_t *tensor) {
    if (tensor->cl_mem != NULL) return;
    tensor->cl_mem = malloc(sizeof(cl_mem));
    *((cl_mem *) tensor->cl_mem) = rml_cl_create_buffer(CL_MEM_READ_WRITE, tensor->dims->flat_size * rml_sizeof_type(tensor->tensor_type));
    rml_cl_enqueue_write_buffer(*((cl_mem *) tensor->cl_mem), tensor->dims->flat_size * rml_sizeof_type(tensor->tensor_type), tensor->data);
}

void rml_cl_to_cpu_tensor(tensor_t *tensor) {
    if (tensor->cl_mem == NULL) return;
    rml_cl_enqueue_read_buffer(*((cl_mem *) tensor->cl_mem), tensor->dims->flat_size * rml_sizeof_type(tensor->tensor_type), tensor->data);
    free(tensor->cl_mem);
    tensor->cl_mem = NULL;
}

cl_mem rml_cl_create_buffer(int mem_properties, size_t size) {
    return clCreateBuffer(context, mem_properties, size, NULL, NULL);
}

void rml_cl_enqueue_read_buffer(cl_mem buffer, size_t size, void *data) {
    clEnqueueReadBuffer(command_queue, buffer, CL_TRUE, 0, size, data, 0, NULL, NULL);
}

void rml_cl_enqueue_write_buffer(cl_mem buffer, size_t size, void *data) {
    clEnqueueWriteBuffer(command_queue, buffer, CL_TRUE, 0, size, data, 0, NULL, NULL);
}

void rml_cl_set_kernel_arg(op_code_t op_code, tensor_type_t tensor_type, size_t arg_index, cl_mem *buffer) {
    clSetKernelArg(kernels[op_code][tensor_type], arg_index, sizeof(cl_mem), buffer);
}

void rml_cl_enqueue_range_kernel(op_code_t op_code, tensor_type_t tensor_type, size_t op_size) {
    clEnqueueNDRangeKernel(command_queue, kernels[op_code][tensor_type], 1, NULL, &op_size, NULL, 0, NULL, NULL);
}

void rml_cl_finish() {
    clFinish(command_queue);
}

void rml_cl_free_buffer(cl_mem buffer) {
    clReleaseMemObject(buffer);
}

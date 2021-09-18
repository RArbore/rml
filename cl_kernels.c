#include "cl_kernels.h"

cl_context context;
cl_command_queue command_queue;

const char *rml_cl_program =
"__kernel void addf(__global float *a, __global float *b, __global float *c)\n"\
"{\n"\
"  size_t id = get_global_id(0);\n"\
"  c[id] = a[id] + b[id];\n"\
"}\n"\
"\n";

cl_kernel kernels[NUM_OP_CODES][NUM_TYPES];

void rml_cl_kernel_init() {
    cl_int err;
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) &rml_cl_program, NULL, &err);
    if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
        printf("Error building cl program\n");
    }

    kernels[OP_CODE_ADD][TENSOR_TYPE_FLOAT] = clCreateKernel(program, "addf", &err);
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

    rml_cl_kernel_init();
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
    clFinish(command_queue);
}

void rml_cl_free_buffer(cl_mem buffer) {
    clReleaseMemObject(buffer);
}

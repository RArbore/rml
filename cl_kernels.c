#include "cl_kernels.h"

const char *rml_cl_addf =
"__kernel void add(__global float *a, __global float *b, __global float *result) {\n"
"    size_t id = get_global_id(0);\n"
"    output[id] = a[id] + b[id];\n"
"}\n";

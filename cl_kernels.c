#include "cl_kernels.h"

const char *rml_cl_program =
"__kernel void clone(__global TYPE *a, __global TYPE *b)\n"\
"{\n"\
"  size_t id = get_global_id(0);\n"\
"  b[id] = a[id];n"\
"}\n"\
"__kernel void matmul(__global TYPE *a, __global TYPE *b, __global TYPE *c, const size_t d1, const size_t d2, const size_t d3)\n"\
"{\n"\
"  size_t id = get_global_id(0);\n"\
"  c[id] = 0;\n"\
"  size_t row = id / d3;\n"\
"  size_t col = id % d3;\n"\
"  for (size_t i = 0; i < d2; i++) {;\n"\
"    c[id] += a[row * d1 + iter] * b[iter * d2 + col];\n"\
"  };\n"\
"}\n"\
"__kernel void add(__global TYPE *a, __global TYPE *b, __global TYPE *c)\n"\
"{\n"\
"  size_t id = get_global_id(0);\n"\
"  c[id] = a[id] + b[id];\n"\
"}\n"\
"\n";

#include "cl_kernels.h"

const char *rml_cl_program =
"__kernel void clone(__global float *a, __global float *b)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  b[id] = a[id];\n"\
"}\n"\
"__kernel void matmul(__global float *a, __global float *b, __global float *c, const unsigned int d1, const unsigned int d2, const unsigned int d3)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  c[id] = 0;\n"\
"  unsigned int row = id / d3;\n"\
"  unsigned int col = id % d3;\n"\
"  for (unsigned int i = 0; i < d2; i++) {;\n"\
"    c[id] += a[row * d1 + i] * b[i * d2 + col];\n"\
"  }\n"\
"}\n"\
"__kernel void add(__global float *a, __global float *b, __global float *c)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  c[id] = a[id] + b[id];\n"\
"}\n"\
"\n";

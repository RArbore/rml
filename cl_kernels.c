#include "cl_kernels.h"

const char *rml_cl_program =
"__kernel void clone(__global TYPE *a, __global TYPE *b)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  b[id] = a[id];\n"\
"}\n"\
"__kernel void matmul(__global TYPE *a, __global TYPE *b, __global TYPE *c, const unsigned int d1, const unsigned int d2, const unsigned int d3)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  c[id] = 0;\n"\
"  unsigned int row = id / d3;\n"\
"  unsigned int col = id % d3;\n"\
"  for (unsigned int i = 0; i < d2; i++) {;\n"\
"    c[id] += a[row * d1 + i] * b[i * d2 + col];\n"\
"  }\n"\
"}\n"\
"__kernel void concat(__global TYPE *a, __global TYPE *b, __global TYPE *c, const unsigned int pitch_a, const unsigned int pitch_b)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  unsigned int total_pitch = pitch_a + pitch_b;\n"\
"  if (id % total_pitch < pitch_a) c[id] = a[id];\n"\
"  else c[id] = b[id];\n"\
"}\n"\
"__kernel void slice(__global TYPE *tensor, __global TYPE *result, __global unsigned int *lower_bound, __global unsigned int *upper_bound, __global unsigned int *dims, const unsigned int num_dims)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  unsigned int pos_workspace[num_dims], i_divided = id;\n"\
"  int reached_zero = 0;\n"\
"  for (unsigned int d = num_dims - 1; !reached_zero; d--) {\n"\
"    if (d < num_dims - 1) i_divided /= dims[d + 1];\n"\
"    pos_workspace[d] = i_divided % dims[d] + lower_bound[d];\n"\
"    if (d == 0) reached_zero = 1;\n"\
"  }\n"\
"  unsigned int old_pos = 0;\n"\
"  for (unsigned int d = 0; d < num_dims; d++) {\n"\
"    unsigned int prev_mult = 0;\n"\
"    if (d > 0) prev_mult = dims[d];\n"\
"    old_pos = old_pos * prev_mult + pos_workspace[d];\n"\
"  }\n"\
"  result[i] = tensor[old_pos];\n"\
"}\n"\
"__kernel void slice(__global TYPE *tensor, __global TYPE *result, __global unsigned int *lower_bound, __global unsigned int *dims, const unsigned int num_dims)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  unsigned int pos_workspace[num_dims], i_divided = id;\n"\
"  int reached_zero = 0;\n"\
"  for (unsigned int d = num_dims - 1; !reached_zero; d--) {\n"\
"    if (d < num_dims - 1) i_divided /= dims[d + 1];\n"\
"    pos_workspace[d] = i_divided % dims[d] + lower_bound[d];\n"\
"    if (d == 0) reached_zero = 1;\n"\
"  }\n"\
"  unsigned int new_pos = 0;\n"\
"  for (unsigned int d = 0; d < num_dims; d++) {\n"\
"    unsigned int prev_mult = 0;\n"\
"    if (d > 0) prev_mult = dims[d];\n"\
"    new_pos = new_pos * prev_mult + pos_workspace[d];\n"\
"  }\n"\
"  result[new_pos] = tensor[id]\n"\
"  \n"\
"}\n"\
"__kernel void add(__global TYPE *a, __global TYPE *b, __global TYPE *c)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  c[id] = a[id] + b[id];\n"\
"}\n"\
"\n";

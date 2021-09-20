#include "cl_kernels.h"

const char *rml_cl_program =
"__kernel void clone(__global TYPE *a, __global TYPE *b)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  b[id] = a[id];\n"\
"}\n"\
"\n"\
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
"\n"\
"__kernel void concat(__global TYPE *a, __global TYPE *b, __global TYPE *c, const unsigned int pitch_a, const unsigned int pitch_b)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  unsigned int total_pitch = pitch_a + pitch_b;\n"\
"  if (id % total_pitch < pitch_a) c[id] = a[id];\n"\
"  else c[id] = b[id];\n"\
"}\n"\
"\n"\
"__kernel void slice(__global TYPE *tensor, __global TYPE *result, __global unsigned int *lower_bound, __global unsigned int *upper_bound, __global unsigned int *dims, const unsigned int num_dims)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  unsigned int pos_workspace[MAX_ARR_SIZE];\n"\
"  unsigned int i_divided = id;\n"\
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
"  result[id] = tensor[old_pos];\n"\
"}\n"\
"\n"\
"__kernel void assign_slice(__global TYPE *tensor, __global TYPE *result, __global unsigned int *lower_bound, __global unsigned int *dims, const unsigned int num_dims)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  unsigned int pos_workspace[MAX_ARR_SIZE];\n"\
"  unsigned int i_divided = id;\n"\
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
"  result[new_pos] = tensor[id];\n"\
"}\n"\
"\n"\
"__kernel void transpose(__global TYPE *tensor, __global TYPE *result, const unsigned int in_r, const unsigned int in_c)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  unsigned int new_r = id / in_r\n"\
"  unsigned int new_c = id % in_r\n"\
"  result[id] = tensor[new_c * in_c + new_r];\n"\
"}\n"\
"\n"\
"__kernel void permute(__global TYPE *tensor, __global TYPE *result, __global unsigned int *perms, __global unsigned int *dims, const unsigned int num_dims)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  unsigned int pos_workspace[MAX_ARR_SIZE];\n"\
"  unsigned int i_divided = id;\n"\
"  int reached_zero = 0;\n"\
"  for (unsigned int d = num_dims - 1; !reached_zero; d--) {\n"\
"    if (d < num_dims - 1) i_divided /= dims[d + 1];\n"\
"    pos_workspace[d] = i_divided % dims[d];\n"\
"    if (d == 0) reached_zero = 1;\n"\
"  }\n"\
"  unsigned int new_pos = 0;\n"\
"  for (unsigned int d = 0; d < num_dims; d++) {\n"\
"    unsigned int prev_mult = 0;\n"\
"    if (d > 0) prev_mult = dims[perms[d]];\n"\
"    new_pos = new_pos * prev_mult + pos_workspace[perms[d]];\n"\
"  }\n"\
"  result[new_pos] = tensor[id];\n"\
"}\n"\
"\n"\
"__kernel void add(__global TYPE *a, __global TYPE *b, __global TYPE *c)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  c[id] = a[id] + b[id];\n"\
"}\n"\
"\n"\
"__kernel void sub(__global TYPE *a, __global TYPE *b, __global TYPE *c)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  c[id] = a[id] - b[id];\n"\
"}\n"\
"__kernel void mul(__global TYPE *a, __global TYPE *b, __global TYPE *c)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  c[id] = a[id] * b[id];\n"\
"}\n"\
"\n"\
"__kernel void div(__global TYPE *a, __global TYPE *b, __global TYPE *c)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  c[id] = a[id] / b[id];\n"\
"}\n"\
"\n"\
"__kernel void increment(__global TYPE *a, const TYPE scalar)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] += scalar;\n"\
"}\n"\
"\n"\
"__kernel void scale(__global TYPE *a, const TYPE scalar)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] *= scalar;\n"\
"}\n"\
"\n"\
"__kernel void exp(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = exp(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void log(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = log(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void pow(__global TYPE *a, const TYPE scalar)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = pow(a[id], scalar);\n"\
"}\n"\
"\n"\
"__kernel void sin(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = sin(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void cos(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = cos(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void tan(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = tan(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void sinh(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = sinh(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void cosh(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = cosh(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void tanh(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = tanh(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void asin(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = asin(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void acos(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = acos(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void atan(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = atan(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void asinh(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = asinh(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void acosh(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = acosh(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void atanh(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = atanh(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void abs(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = abs(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void clamp(__global TYPE *a, const TYPE min, const TYPE max, const int code)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  if (code % 2 == 0 && a[id] < min) a[id] = min;\n"\
"  if (code > 0 && a[id] > max) a[id] = max;\n"\
"}\n"\
"\n";
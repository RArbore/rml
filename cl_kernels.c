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

#include "cl_kernels.h"

const char *rml_cl_program =
"__kernel void rml_matmul(__global TYPE *a, __global TYPE *b, __global TYPE *c, const unsigned int d1, const unsigned int d2, const unsigned int d3)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  c[id] = 0;\n"\
"  unsigned int row = id / d3;\n"\
"  unsigned int col = id % d3;\n"\
"  for (unsigned int i = 0; i < d2; i++) {;\n"\
"    c[id] += a[row * d2 + i] * b[i * d3 + col];\n"\
"  }\n"\
"}\n"\
"\n"\
"__kernel void rml_concat(__global TYPE *a, __global TYPE *b, __global TYPE *c, __constant const unsigned int *dims_a, __constant const unsigned int *dims_b, __constant const unsigned int *dims_c, const unsigned int num_dims, const unsigned int cdim)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  unsigned int pos_workspace[MAX_ARR_SIZE];\n"\
"  unsigned int i_divided = id;\n"\
"  int reached_zero = 0;\n"\
"  for (unsigned int d = num_dims - 1; !reached_zero; d--) {\n"\
"    if (d < num_dims - 1) i_divided /= dims_c[d + 1];\n"\
"    pos_workspace[d] = i_divided % dims_c[d];\n"\
"    if (d == 0) reached_zero = 1;\n"\
"  }\n"\
"  unsigned int old_pos = 0;\n"\
"  if (pos_workspace[cdim] < dims_a[cdim]) {\n"\
"    for (unsigned int d = 0; d < num_dims; d++) {\n"\
"      unsigned int prev_mult = 0;\n"\
"      if (d > 0) prev_mult = dims_a[d];\n"\
"      old_pos = old_pos * prev_mult + pos_workspace[d];\n"\
"    }\n"\
"    c[id] = a[old_pos];\n"\
"  }\n"\
"  else {\n"\
"    pos_workspace[cdim] -= dims_a[cdim];\n"\
"    for (unsigned int d = 0; d < num_dims; d++) {\n"\
"      unsigned int prev_mult = 0;\n"\
"      if (d > 0) prev_mult = dims_b[d];\n"\
"      old_pos = old_pos * prev_mult + pos_workspace[d];\n"\
"    }\n"\
"    c[id] = b[old_pos];\n"\
"  }\n"\
"  \n"\
"}\n"\
"\n"\
"__kernel void rml_slice(__global TYPE *tensor, __global TYPE *result, __constant const unsigned int *lower_bound, __constant const unsigned int *upper_bound, __constant const unsigned int *tensor_dims, __constant const unsigned int *result_dims, const unsigned int num_dims)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  unsigned int pos_workspace[MAX_ARR_SIZE];\n"\
"  unsigned int i_divided = id;\n"\
"  int reached_zero = 0;\n"\
"  for (unsigned int d = num_dims - 1; !reached_zero; d--) {\n"\
"    if (d < num_dims - 1) i_divided /= result_dims[d + 1];\n"\
"    pos_workspace[d] = i_divided % result_dims[d] + lower_bound[d];\n"\
"    if (d == 0) reached_zero = 1;\n"\
"  }\n"\
"  unsigned int old_pos = 0;\n"\
"  for (unsigned int d = 0; d < num_dims; d++) {\n"\
"    unsigned int prev_mult = 0;\n"\
"    if (d > 0) prev_mult = tensor_dims[d];\n"\
"    old_pos = old_pos * prev_mult + pos_workspace[d];\n"\
"  }\n"\
"  result[id] = tensor[old_pos];\n"\
"}\n"\
"\n"\
"__kernel void rml_assign_slice(__global TYPE *tensor, __global TYPE *result, __constant unsigned int *lower_bound, __constant unsigned int *tensor_dims, __constant unsigned int *result_dims, const unsigned int num_dims)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  unsigned int pos_workspace[MAX_ARR_SIZE];\n"\
"  unsigned int i_divided = id;\n"\
"  int reached_zero = 0;\n"\
"  for (unsigned int d = num_dims - 1; !reached_zero; d--) {\n"\
"    if (d < num_dims - 1) i_divided /= tensor_dims[d + 1];\n"\
"    pos_workspace[d] = i_divided % tensor_dims[d] + lower_bound[d];\n"\
"    if (d == 0) reached_zero = 1;\n"\
"  }\n"\
"  unsigned int new_pos = 0;\n"\
"  for (unsigned int d = 0; d < num_dims; d++) {\n"\
"    unsigned int prev_mult = 0;\n"\
"    if (d > 0) prev_mult = result_dims[d];\n"\
"    new_pos = new_pos * prev_mult + pos_workspace[d];\n"\
"  }\n"\
"  result[new_pos] = tensor[id];\n"\
"}\n"\
"\n"\
"__kernel void rml_transpose(__global TYPE *tensor, __global TYPE *result, const unsigned int in_r, const unsigned int in_c)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  unsigned int new_r = id / in_r;\n"\
"  unsigned int new_c = id % in_r;\n"\
"  result[id] = tensor[new_c * in_c + new_r];\n"\
"}\n"\
"\n"\
"__kernel void rml_permute(__global TYPE *tensor, __global TYPE *result, __constant unsigned int *perms, __constant unsigned int *dims, const unsigned int num_dims)\n"\
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
"__kernel void rml_cast_float(__global TYPE *a, __global float *b)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  b[id] = (float) a[id];\n"\
"}\n"\
"\n"\
"__kernel void rml_cast_double(__global TYPE *a, __global double *b)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  b[id] = (double) a[id];\n"\
"}\n"\
"\n"\
"__kernel void rml_add(__global TYPE *a, __global TYPE *b, __global TYPE *c)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  c[id] = a[id] + b[id];\n"\
"}\n"\
"\n"\
"__kernel void rml_sub(__global TYPE *a, __global TYPE *b, __global TYPE *c)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  c[id] = a[id] - b[id];\n"\
"}\n"\
"__kernel void rml_mul(__global TYPE *a, __global TYPE *b, __global TYPE *c)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  c[id] = a[id] * b[id];\n"\
"}\n"\
"\n"\
"__kernel void rml_div(__global TYPE *a, __global TYPE *b, __global TYPE *c)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  c[id] = a[id] / b[id];\n"\
"}\n"\
"\n"\
"__kernel void rml_increment(__global TYPE *a, const TYPE scalar)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] += scalar;\n"\
"}\n"\
"\n"\
"__kernel void rml_scale(__global TYPE *a, const TYPE scalar)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] *= scalar;\n"\
"}\n"\
"\n"\
"__kernel void rml_exp(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = exp(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void rml_log(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = log(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void rml_pow(__global TYPE *a, const TYPE scalar)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = pow(a[id], scalar);\n"\
"}\n"\
"\n"\
"__kernel void rml_sin(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = sin(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void rml_cos(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = cos(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void rml_tan(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = tan(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void rml_sinh(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = sinh(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void rml_cosh(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = cosh(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void rml_tanh(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = tanh(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void rml_asin(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = asin(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void rml_acos(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = acos(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void rml_atan(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = atan(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void rml_asinh(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = asinh(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void rml_acosh(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = acosh(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void rml_atanh(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = atanh(a[id]);\n"\
"}\n"\
"\n"\
"__kernel void rml_abs(__global TYPE *a)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  a[id] = a[id] >= 0 ? a[id] : -a[id];\n"\
"}\n"\
"\n"\
"__kernel void rml_clamp(__global TYPE *a, const TYPE min, const TYPE max, const unsigned int code)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  if (code % 2 == 0 && a[id] < min) a[id] = min;\n"\
"  if (code > 0 && a[id] > max) a[id] = max;\n"\
"}\n"\
"\n"\
"__kernel void rml_max(__global TYPE *tensor, __global TYPE *result, const unsigned int pool_size, const unsigned int in_size)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  for (unsigned int i = 0; i < pool_size && id * pool_size + i < in_size; i++) {\n"\
"    if (i == 0 || result[id] < tensor[id * pool_size + i]) result[id] = tensor[id * pool_size + i];\n"\
"  }\n"\
"  \n"\
"}\n"\
"\n"\
"__kernel void rml_min(__global TYPE *tensor, __global TYPE *result, const unsigned int pool_size, const unsigned int in_size)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  for (unsigned int i = 0; i < pool_size && id * pool_size + i < in_size; i++) {\n"\
"    if (i == 0 || result[id] > tensor[id * pool_size + i]) result[id] = tensor[id * pool_size + i];\n"\
"  }\n"\
"}\n"\
"\n"\
"__kernel void rml_sum(__global TYPE *tensor, __global TYPE *result, const unsigned int pool_size, const unsigned int in_size)\n"\
"{\n"\
"  unsigned int id = get_global_id(0);\n"\
"  for (unsigned int i = 0; i < pool_size && id * pool_size + i < in_size; i++) {\n"\
"    if (i == 0) result[id] = tensor[id * pool_size + i];\n"\
"    else result[id] += tensor[id * pool_size + i];\n"\
"  }\n"\
"}\n"\
"\n";

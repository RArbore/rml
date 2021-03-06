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

#ifndef INTERNAL_H_
#define INTERNAL_H_

#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>

#include "rml.h"

#define SS(type) (sysconf(_SC_LEVEL1_DCACHE_LINESIZE) / sizeof(type))

#define INCREMENT_VOID_POINTER_VAL(type, ptr, i, val) {*((type *) ptr + i) += val;}
#define SCALE_VOID_POINTER_VAL(type, ptr, i, val) {*((type *) ptr + i) *= val;}
#define INCREMENT_VOID_POINTER_PTR(type, ptr, i, ptr_val, i_val) {*((type *) ptr + i) += *((type *) ptr_val + i_val);}
#define SCALE_VOID_POINTER_PTR(type, ptr, i, ptr_val, i_val) {*((type *) ptr + i) *= *((type *) ptr_val + i_val);}
#define INV_VOID_POINTER(type, ptr, i) {*((type *) ptr + i) = 1 / *((type *) ptr + i);}

#define ASSIGN_VOID_POINTER(type, dest, value, index) {*((type *) dest + index) = (type) value;}
#define MALLOC_VOID_POINTER(type, ptr, size) {ptr = malloc(size * sizeof(type));}
#define CALLOC_VOID_POINTER(type, ptr, size) {ptr = calloc(size, sizeof(type));}
#define CAST_VOID_POINTER(type_new, type_old, dest, src, i1, i2) {*((type_new *) dest + i1) = *((type_old *) src + i2);}
#define CAST_VOID_POINTER_SWAPPED_TYPES(type_old, type_new, dest, src, i1, i2) {*((type_new *) dest + i1) = *((type_old *) src + i2);}
#define COPY_VOID_POINTER(type, dest, src, i1, i2) CAST_VOID_POINTER(type, type, dest, src, i1, i2);
#define STORE_VOID_FROM_VA(type_dest, type_va, ap, dest, index) {*((type_dest *) dest + index) = va_arg(ap, type_va);}
#define ABS_VOID_POINTER(type, dest, src, i1, i2) {*((type *) dest + i1) = *((type *) src + i2) >= 0 ? *((type *) src + i2) : -*((type *) src + i2);}
#define ACCUM_VOID_POINTER(type, a, b, c, i1, i2, i3) {*((type *) c + i3) += *((type *) a + i1) * *((type *) b + i2);}
#define COMPARE_VOID_POINTER(type, a, b, i1, i2, res) { \
        if(*((type *) a + i1) > *((type *) b + i2)) res = 1; \
        else if(*((type *) a + i1) < *((type *) b + i2)) res = -1; \
        else res = 0; \
    }
#define IS_ZERO_VOID_POINTER(type, ptr, i, dest) {dest = *((type *) ptr + i) == 0;}

#define FAST_MATRIX_MULTIPLY(type, a, b_clone, result) { \
        type *a_data = (type *) a->data; \
        type *b_clone_data = (type *) b_clone->data; \
        type *result_data = (type *) result->data; \
        size_t index_res = 0; \
        size_t inner_dim = a->dims->dims[1]; \
        for (size_t r = 0; r < result->dims->dims[0]; r++) { \
            for (size_t c = 0; c < result->dims->dims[1]; c++) { \
                size_t index_a = r * inner_dim; \
                size_t index_b = c * inner_dim; \
                for (size_t i = 0; i < inner_dim; i++) { \
                    result_data[index_res] += a_data[index_a++] * b_clone_data[index_b++]; \
                } \
                index_res++; \
            } \
        } \
    }

/*
#define FAST_MATRIX_MULTIPLY(type, a, b, result) { \
        type *a_data = (type *) a->data; \
        type *b_data = (type *) b->data; \
        type *result_data = (type *) result->data; \
        size_t inner_dim = a->dims->dims[1]; \
        size_t left_dim = result->dims->dims[0]; \
        size_t right_dim = result->dims->dims[1]; \
        size_t ss = SS(type); \
        type *res_ptr, *a_ptr, *b_ptr; \
        size_t r, c, i, r2, c2, i2; \
        for (r = 0; r < left_dim; r += ss) { \
            for (c = 0; c < right_dim; c += ss) { \
                for (i = 0; i < inner_dim; i += ss) { \
                    for (r2 = 0, \
                        res_ptr = result_data + r * right_dim + c, \
                        a_ptr = a_data + r * inner_dim + i; \
                        r2 < ss; r2++, \
                        res_ptr += right_dim, a_ptr += inner_dim) { \
                        for (i2 = 0, \
                            b_ptr = b_data + i * right_dim + c; \
                            i2 < ss; i2++, \
                            b_ptr += right_dim) { \
                            for (c2 = 0; c2 < ss; c2++) { \
                                res_ptr[c2] += a_ptr[i2] * b_ptr[c2]; \
                            } \
                        } \
                    } \
                } \
            } \
        } \
    }
*/

#define CAST_TENSORS_WIDEN(a, b) \
    tensor_t *A_CASTED = NULL; \
    tensor_t *B_CASTED = NULL; \
    { \
        if (a->tensor_type > b->tensor_type) { \
            b = rml_cast_tensor(b, a->tensor_type); \
            B_CASTED = b; \
        } \
        else if (a->tensor_type < b->tensor_type) { \
            a = rml_cast_tensor(a, b->tensor_type); \
            A_CASTED = a; \
        } \
        assert(a->tensor_type == b->tensor_type); \
    }

#define CLEANUP_CAST_TENSORS_WIDEN { \
        rml_free_tensor(A_CASTED); \
        rml_free_tensor(B_CASTED); \
    }

#define ADD_TENSORS(type, tensor_a, tensor_b, tensor_c) { \
        type *cast_a = (type *) tensor_a->data; \
        type *cast_b = (type *) tensor_b->data; \
        type *cast_c = (type *) tensor_c->data; \
        for (size_t i = 0; i < tensor_a->dims->flat_size; i++) { \
            cast_c[i] = cast_a[i] + cast_b[i]; \
        } \
    }

#define SUB_TENSORS(type, tensor_a, tensor_b, tensor_c) { \
        type *cast_a = (type *) tensor_a->data; \
        type *cast_b = (type *) tensor_b->data; \
        type *cast_c = (type *) tensor_c->data; \
        for (size_t i = 0; i < tensor_a->dims->flat_size; i++) { \
            cast_c[i] = cast_a[i] - cast_b[i]; \
        } \
    }

#define MUL_TENSORS(type, tensor_a, tensor_b, tensor_c) { \
        type *cast_a = (type *) tensor_a->data; \
        type *cast_b = (type *) tensor_b->data; \
        type *cast_c = (type *) tensor_c->data; \
        for (size_t i = 0; i < tensor_a->dims->flat_size; i++) { \
            cast_c[i] = cast_a[i] * cast_b[i]; \
        } \
    }

#define DIV_TENSORS(type, tensor_a, tensor_b, tensor_c) { \
        type *cast_a = (type *) tensor_a->data; \
        type *cast_b = (type *) tensor_b->data; \
        type *cast_c = (type *) tensor_c->data; \
        for (size_t i = 0; i < tensor_a->dims->flat_size; i++) { \
            cast_c[i] = cast_a[i] / cast_b[i]; \
        } \
    }

#define INCREMENT_TENSOR(type, tensor_src, scalar, tensor_dest) { \
        type *cast_src = (type *) tensor_src->data; \
        type *cast_scalar = (type *) scalar; \
        type *cast_dest = (type *) tensor_dest->data; \
        for (size_t i = 0; i < tensor_src->dims->flat_size; i++) { \
            cast_dest[i] = cast_src[i] + *cast_scalar; \
        } \
    }

#define SCALE_TENSOR(type, tensor_src, scalar, tensor_dest) { \
        type *cast_src = (type *) tensor_src->data; \
        type *cast_scalar = (type *) scalar; \
        type *cast_dest = (type *) tensor_dest->data; \
        for (size_t i = 0; i < tensor_src->dims->flat_size; i++) { \
            cast_dest[i] = cast_src[i] * *cast_scalar; \
        } \
    }

#define CLAMP_TENSOR(type, tensor_src, min, max, tensor_dest) { \
        type *cast_src = (type *) tensor_src->data; \
        type *cast_min = (type *) min; \
        type *cast_max = (type *) max; \
        type *cast_dest = (type *) tensor_dest->data; \
        if (min != NULL && max != NULL) { \
            for (size_t i = 0; i < tensor_src->dims->flat_size; i++) { \
                cast_dest[i] = *cast_min > cast_src[i] ? *cast_min : (*cast_max < cast_src[i] ? *cast_max : cast_src[i]); \
            } \
        } \
        else if (max != NULL) { \
            for (size_t i = 0; i < tensor_src->dims->flat_size; i++) { \
                cast_dest[i] = *cast_max < cast_src[i] ? *cast_max : cast_src[i]; \
            } \
        } \
        else if (min != NULL) { \
            for (size_t i = 0; i < tensor_src->dims->flat_size; i++) { \
                cast_dest[i] = *cast_min > cast_src[i] ? *cast_min : cast_src[i]; \
            } \
        } \
        else { \
            for (size_t i = 0; i < tensor_src->dims->flat_size; i++) { \
                cast_dest[i] = cast_src[i]; \
            } \
        } \
    }

#define SWITCH_ENUM_TYPES(type, macro, ...) \
    switch (type) { \
        case TENSOR_TYPE_BYTE: \
            macro(char, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_UBYTE: \
            macro(unsigned char, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_SHORT: \
            macro(short, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_USHORT: \
            macro(unsigned short, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_INT: \
            macro(int, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_UINT: \
            macro(unsigned int, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_LONG: \
            macro(long, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_ULONG: \
            macro(unsigned long, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_FLOAT: \
            macro(float, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_DOUBLE: \
            macro(double, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_LDOUBLE: \
            macro(long double, ##__VA_ARGS__); \
            break; \
    }

#define SWITCH_2_ENUM_TYPES(type1, type2, macro, ...) \
    switch (type2) { \
        case TENSOR_TYPE_BYTE: \
            SWITCH_ENUM_TYPES(type1, macro, char, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_UBYTE: \
            SWITCH_ENUM_TYPES(type1, macro, unsigned char, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_SHORT: \
            SWITCH_ENUM_TYPES(type1, macro, short, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_USHORT: \
            SWITCH_ENUM_TYPES(type1, macro, unsigned short, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_INT: \
            SWITCH_ENUM_TYPES(type1, macro, int, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_UINT: \
            SWITCH_ENUM_TYPES(type1, macro, unsigned int, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_LONG: \
            SWITCH_ENUM_TYPES(type1, macro, long, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_ULONG: \
            SWITCH_ENUM_TYPES(type1, macro, unsigned long, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_FLOAT: \
            SWITCH_ENUM_TYPES(type1, macro, float, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_DOUBLE: \
            SWITCH_ENUM_TYPES(type1, macro, double, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_LDOUBLE: \
            SWITCH_ENUM_TYPES(type1, macro, long double, ##__VA_ARGS__); \
            break; \
    }

#define SWITCH_ENUM_TYPES_VA(type, ...) \
    switch (type) { \
        case TENSOR_TYPE_BYTE: \
            STORE_VOID_FROM_VA(char, int, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_UBYTE: \
            STORE_VOID_FROM_VA(unsigned char, int, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_SHORT: \
            STORE_VOID_FROM_VA(short, int, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_USHORT: \
            STORE_VOID_FROM_VA(unsigned short, int, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_INT: \
            STORE_VOID_FROM_VA(int, int, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_UINT: \
            STORE_VOID_FROM_VA(unsigned int, unsigned int, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_LONG: \
            STORE_VOID_FROM_VA(long, long, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_ULONG: \
            STORE_VOID_FROM_VA(unsigned long, unsigned long, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_FLOAT: \
            STORE_VOID_FROM_VA(float, double, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_DOUBLE: \
            STORE_VOID_FROM_VA(double, double, ##__VA_ARGS__); \
            break; \
        case TENSOR_TYPE_LDOUBLE: \
            STORE_VOID_FROM_VA(long double, long double, ##__VA_ARGS__); \
            break; \
    }

#define INDEX_VOID_POINTER(type, ptr, index) (\
    type == TENSOR_TYPE_BYTE ? (void *) ((char *) ptr + index) : \
    type == TENSOR_TYPE_UBYTE ? (void *) ((unsigned char *) ptr + index) : \
    type == TENSOR_TYPE_SHORT ? (void *) ((short *) ptr + index) : \
    type == TENSOR_TYPE_USHORT ? (void *) ((unsigned short *) ptr + index) : \
    type == TENSOR_TYPE_INT ? (void *) ((int *) ptr + index) : \
    type == TENSOR_TYPE_UINT ? (void *) ((unsigned int *) ptr + index) : \
    type == TENSOR_TYPE_LONG ? (void *) ((long *) ptr + index) : \
    type == TENSOR_TYPE_ULONG ? (void *) ((unsigned long *) ptr + index) : \
    type == TENSOR_TYPE_FLOAT ? (void *) ((float *) ptr + index) : \
    type == TENSOR_TYPE_DOUBLE ? (void *) ((double *) ptr + index) : \
    (void *) ((long double *) ptr + index) \
)

#define PRINT_VOID_POINTER(type, ptr, index) \
    switch (type) { \
        case TENSOR_TYPE_BYTE: \
            printf("%d", *((char *) ptr + index)); \
            break; \
        case TENSOR_TYPE_UBYTE: \
            printf("%u", *((unsigned char *) ptr + index)); \
            break; \
        case TENSOR_TYPE_SHORT: \
            printf("%d", *((short *) ptr + index)); \
            break; \
        case TENSOR_TYPE_USHORT: \
            printf("%u", *((unsigned short *) ptr + index)); \
            break; \
        case TENSOR_TYPE_INT: \
            printf("%d", *((int *) ptr + index)); \
            break; \
        case TENSOR_TYPE_UINT: \
            printf("%u", *((unsigned int *) ptr + index)); \
            break; \
        case TENSOR_TYPE_LONG: \
            printf("%ld", *((long *) ptr + index)); \
            break; \
        case TENSOR_TYPE_ULONG: \
            printf("%lu", *((unsigned long *) ptr + index)); \
            break; \
        case TENSOR_TYPE_FLOAT: \
            printf("%f", *((float *) ptr + index)); \
            break; \
        case TENSOR_TYPE_DOUBLE: \
            printf("%f", *((double *) ptr + index)); \
            break; \
        case TENSOR_TYPE_LDOUBLE: \
            printf("%Lf", *((long double *) ptr + index)); \
            break; \
    }

#define FPRINT_VOID_POINTER(type, ptr, index, fp) \
    switch (type) { \
        case TENSOR_TYPE_BYTE: \
            fprintf(fp, "%d", *((char *) ptr + index)); \
            break; \
        case TENSOR_TYPE_UBYTE: \
            fprintf(fp, "%u", *((unsigned char *) ptr + index)); \
            break; \
        case TENSOR_TYPE_SHORT: \
            fprintf(fp, "%d", *((short *) ptr + index)); \
            break; \
        case TENSOR_TYPE_USHORT: \
            fprintf(fp, "%u", *((unsigned short *) ptr + index)); \
            break; \
        case TENSOR_TYPE_INT: \
            fprintf(fp, "%d", *((int *) ptr + index)); \
            break; \
        case TENSOR_TYPE_UINT: \
            fprintf(fp, "%u", *((unsigned int *) ptr + index)); \
            break; \
        case TENSOR_TYPE_LONG: \
            fprintf(fp, "%ld", *((long *) ptr + index)); \
            break; \
        case TENSOR_TYPE_ULONG: \
            fprintf(fp, "%lu", *((unsigned long *) ptr + index)); \
            break; \
        case TENSOR_TYPE_FLOAT: \
            fprintf(fp, "%f", *((float *) ptr + index)); \
            break; \
        case TENSOR_TYPE_DOUBLE: \
            fprintf(fp, "%f", *((double *) ptr + index)); \
            break; \
        case TENSOR_TYPE_LDOUBLE: \
            fprintf(fp, "%Lf", *((long double *) ptr + index)); \
            break; \
    }

#define FSCANF_VOID_POINTER(type, ptr, index, fp) \
    switch (type) { \
        case TENSOR_TYPE_BYTE: \
            fscanf(fp, "%c", (char *) ptr + index); \
            break; \
        case TENSOR_TYPE_UBYTE: \
            fscanf(fp, "%c", (unsigned char *) ptr + index); \
            break; \
        case TENSOR_TYPE_SHORT: \
            fscanf(fp, "%hd", (short *) ptr + index); \
            break; \
        case TENSOR_TYPE_USHORT: \
            fscanf(fp, "%hu", (unsigned short *) ptr + index); \
            break; \
        case TENSOR_TYPE_INT: \
            fscanf(fp, "%d", (int *) ptr + index); \
            break; \
        case TENSOR_TYPE_UINT: \
            fscanf(fp, "%u", (unsigned int *) ptr + index); \
            break; \
        case TENSOR_TYPE_LONG: \
            fscanf(fp, "%ld", (long *) ptr + index); \
            break; \
        case TENSOR_TYPE_ULONG: \
            fscanf(fp, "%lu", (unsigned long *) ptr + index); \
            break; \
        case TENSOR_TYPE_FLOAT: \
            fscanf(fp, "%f", (float *) ptr + index); \
            break; \
        case TENSOR_TYPE_DOUBLE: \
            fscanf(fp, "%lf", (double *) ptr + index); \
            break; \
        case TENSOR_TYPE_LDOUBLE: \
            fscanf(fp, "%Lf", (long double *) ptr + index); \
            break; \
    }

#endif // INTERNAL_H_

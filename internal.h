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
#include <stdio.h>

#include "rml.h"

#define ASSIGN_VOID_POINTER(type, dest, value, index) {*((type *) dest + index) = value;}
#define MALLOC_VOID_POINTER(type, ptr, size) {ptr = malloc(size * sizeof(type));}
#define CAST_VOID_POINTER(type_new, type_old, dest, src, i1, i2) {*((type_new *) dest + i1) = *((type_old *) src + i2);}
#define COPY_VOID_POINTER(type, dest, src, i1, i2) CAST_VOID_POINTER(type, type, dest, src, i1, i2);
#define STORE_VOID_FROM_VA(type_dest, type_va, ap, dest, index) {*((type_dest *) dest + index) = va_arg(ap, type_va);}

#define ADD_VOID_POINTERS(type, a, b, c) {*((type *) c) = *((type *) a) + *((type *) b);}
#define SUB_VOID_POINTERS(type, a, b, c) {*((type *) c) = *((type *) a) - *((type *) b);}
#define MUL_VOID_POINTERS(type, a, b, c) {*((type *) c) = *((type *) a) * *((type *) b);}
#define DIV_VOID_POINTERS(type, a, b, c) {*((type *) c) = *((type *) a) / *((type *) b);}

#define ACCUM_VOID_POINTERS(type, a, b, c, i1, i2, i3) {*((type *) c + i3) += *((type *) a + i1) * *((type *) b + i2);}
#define TRANSPOSED_MATRIX_MULTIPLY(type, a, b_clone, result) { \
        type *a_data = (type *) a->data; \
        type *b_clone_data = (type *) b_clone->data; \
        type *result_data = (type *) result->data; \
        for (size_t r = 0; r < result->dims->dims[0]; r++) { \
            for (size_t c = 0; c < result->dims->dims[1]; c++) { \
                size_t index_res = r * result->dims->dims[1] + c; \
                size_t index_a = r * a->dims->dims[1]; \
                size_t index_b = c * b_clone->dims->dims[1]; \
                for (size_t i = 0; i < a->dims->dims[1]; i++) { \
                    result_data[index_res] += a_data[index_a++] * b_clone_data[index_b++]; \
                } \
            } \
        } \
    }

#define CAST_TENSORS_WIDEN(a, b) { \
        if (a->tensor_type > b->tensor_type) { \
            rml_cast_tensor_inplace(b, a->tensor_type); \
        } \
        else if (a->tensor_type < b->tensor_type) { \
            rml_cast_tensor_inplace(a, b->tensor_type); \
        } \
        assert(a->tensor_type == b->tensor_type); \
    }

#define ADD_TENSORS(type, tensor_a, tensor_b, tensor_c) { \
        type *cast_a = (type *) tensor_a->data; \
        type *cast_b = (type *) tensor_b->data; \
        type *cast_c = (type *) tensor_c->data; \
        for (size_t i = 0; i < tensor_a->dims->flat_size; i++) { \
            cast_c[i] = cast_a[i] + cast_b[i]; \
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

size_t rml_sizeof_type(tensor_type_t tensor_type);

#endif // INTERNAL_H_

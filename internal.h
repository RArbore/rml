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
#include <stdio.h>

#include "rml.h"

#define ASSIGN_VOID_POINTER(type, dest, value) (*((type *) dest) = value)
#define COPY_VOID_POINTER(type, dest, src) (*((type *) dest) = *((type *) src))

#define ADD_VOID_POINTERS(type, a, b, c) (*((type *) c) = *((type *) a) + *((type *) b))
#define SUB_VOID_POINTERS(type, a, b, c) (*((type *) c) = *((type *) a) - *((type *) b))
#define MUL_VOID_POINTERS(type, a, b, c) (*((type *) c) = *((type *) a) * *((type *) b))
#define DIV_VOID_POINTERS(type, a, b, c) (*((type *) c) = *((type *) a) / *((type *) b))

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

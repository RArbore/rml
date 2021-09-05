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

#include "rml.h"

#define ASSIGN_VOID_POINTER(type, dest, value) (*((type *) data) = value)

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

size_t rml_sizeof_type(tensor_type_t tensor_type);

#endif // INTERNAL_H_

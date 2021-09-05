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

#include "internal.h"

size_t rml_sizeof_type(tensor_type_t tensor_type) {
    switch (tensor_type) {
        case TENSOR_TYPE_BYTE:
            return sizeof(char);
        case TENSOR_TYPE_UBYTE:
            return sizeof(unsigned char);
        case TENSOR_TYPE_SHORT:
            return sizeof(short);
        case TENSOR_TYPE_USHORT:
            return sizeof(unsigned short);
        case TENSOR_TYPE_INT:
            return sizeof(int);
        case TENSOR_TYPE_UINT:
            return sizeof(unsigned int);
        case TENSOR_TYPE_LONG:
            return sizeof(long);
        case TENSOR_TYPE_ULONG:
            return sizeof(unsigned long);
        case TENSOR_TYPE_FLOAT:
            return sizeof(float);
        case TENSOR_TYPE_DOUBLE:
            return sizeof(double);
        case TENSOR_TYPE_LDOUBLE:
            return sizeof(long double);
    }
}

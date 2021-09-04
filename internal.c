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

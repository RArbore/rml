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

#include "tensor.h"
#include "rml.h"

dims_t *rml_dims(int count, ...) {

}

void rml_free_dims(dims_t *dims) {
    free(dims->dims);
    free(dims);
}

tensor_t *rml_create_tensor(tensor_type_t type, int count, dims_t *dims){

}

tensor_t *rml_zeros_tensor(tensor_type_t type, int count, dims_t *dims){

}

tensor_t *rml_ones_tensor(tensor_type_t type, int count, dims_t *dims){

}

tensor_t *rml_clone_tensor(tensor_t *tensor){

}

void rml_free_tensor(tensor_t *tensor) {
    rml_free_dims(tensor->dims);
    free(tensor->data);
    free(tensor);
}

void *rml_tensor_primitive_access(tensor_t *tensor, dims_t *dims){

}

tensor_t *rml_tensor_matmul(tensor_t *a, tensor_t *b){

}

tensor_t *rml_cast_tensor_inplace(tensor_t *tensor, tensor_type_t type){

}

tensor_t *rml_tensor_add_inplace(tensor_t *a, tensor_t *b){

}

tensor_t *rml_tensor_mul_inplace(tensor_t *a, tensor_t *b){

}

tensor_t *rml_concat_inplace(tensor_t *a, tensor_t *b, unsigned char dim){

}

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

#include <stdio.h>
#include <time.h>
#include <rml.h>

int main() {
    tensor_t *a = rml_rand_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(3, 10, 10, 10));
    rml_write_tensor_hex("rand.hex", a);
    tensor_t *b = rml_read_tensor_hex("rand.hex", TENSOR_TYPE_FLOAT, rml_create_dims(3, 10, 10, 10));
    tensor_t *c = rml_sub_tensor(a, b);
    tensor_t *sum = rml_sum_tensor(c);
    rml_print_tensor(sum);
}

/*  This file is part of rml.

    rml is rml_free_tensor software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Rml_Free_Tensor Software Foundation, either version 3 of the License, or
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
    float af[] = {0., 1., 2., 3.};
    float bf[] = {4., 5., 6., 7.};

    tensor_t *a = rml_init_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 2, 2), af);
    tensor_t *b = rml_init_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 2, 2), bf);
    tensor_t *c = rml_matmul_tensor(a, b);
    rml_print_tensor(c);
    rml_calc_gradient(c);
    rml_print_tensor(c->jacob_a);
    rml_print_tensor(c->jacob_b);
}

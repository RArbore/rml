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
    rml_cl_init();

    double af[] = {1., 2., 3., 4., 5., 6.};
    double bf[] = {1., 2., 3., 4., 5., 6., 7., 8., 9.};

    for (;;) {
        tensor_t *a = rml_init_tensor(TENSOR_TYPE_DOUBLE, rml_create_dims(2, 2, 3), af);
        tensor_t *b = rml_init_tensor(TENSOR_TYPE_DOUBLE, rml_create_dims(2, 3, 3), bf);
        rml_cpu_to_cl_tensor(a);
        rml_cpu_to_cl_tensor(b);
        tensor_t *c = rml_concat_tensor(a, b, 0);
        rml_cl_to_cpu_tensor(c);
        rml_print_tensor(c);
        rml_free_tensor(a);
        rml_free_tensor(b);
        rml_free_tensor(c);
    }
}

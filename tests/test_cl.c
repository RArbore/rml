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

    float af[] = {1., 2., 3., 4., 5., 6.};
    float bf[] = {6., 5., 4., 3., 2., 1.};

    for (;;) {
        tensor_t *a = rml_init_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 2, 3), af);
        tensor_t *b = rml_init_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 2, 3), bf);
        rml_print_tensor(a);
        rml_print_tensor(b);
        rml_cpu_to_cl_tensor(a);
        rml_cpu_to_cl_tensor(b);
        float two = 2.;
        tensor_t *c = rml_increment_tensor(a, &two);
        tensor_t *d = rml_scale_tensor(a, &two);
        rml_cl_to_cpu_tensor(c);
        rml_cl_to_cpu_tensor(d);
        rml_print_tensor(c);
        rml_print_tensor(d);
        rml_free_tensor(a);
        rml_free_tensor(b);
        rml_free_tensor(c);
        rml_free_tensor(d);
    }
}

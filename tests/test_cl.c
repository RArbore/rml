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

    for (;;) {
        tensor_t *a = rml_init_tensor(TENSOR_TYPE_DOUBLE, rml_create_dims(3, 2, 1, 3), af);
        rml_print_dims(a->dims);
        rml_print_tensor(a);
        rml_cpu_to_cl_tensor(a);
        size_t perms[] = {1, 2, 0};
        tensor_t *b = rml_permute_tensor(a, perms);
        rml_cl_to_cpu_tensor(b);
        rml_print_dims(b->dims);
        rml_print_tensor(b);
        rml_free_tensor(a);
        rml_free_tensor(b);
    }
}

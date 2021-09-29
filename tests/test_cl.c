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

    tensor_t *a = rml_rand_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 100, 100));
    tensor_t *b = rml_rand_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 100, 100));
    rml_cpu_to_cl_tensor(a);
    rml_cpu_to_cl_tensor(b);

    for (;;) {
        time_t seconds;
        seconds = time(NULL);
        for (size_t i = 0; i < 100000; i++) {
            tensor_t *c = rml_matmul_tensor(a, b);
            rml_free_tensor(c);
        }
        seconds = time(NULL) - seconds;
        printf("Seconds taken: %ld\n", seconds);
    }
}

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
    float p[] = {0., 1., 2., 3., 4., 5.};
    float l[] = {0., 0., 0., 1., 0., 0.};
    float pow = 2;

    tensor_t *a = rml_init_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 2, 3), p);
    for (size_t i = 0; i < 1000; i++) {
        rml_set_param_tensor(a);
        tensor_t *label = rml_init_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 2, 3), l);
        tensor_t *b = rml_softmax_tensor(a);
        tensor_t *diff = rml_sub_tensor(b, label);
        tensor_t *power = rml_pow_tensor(diff, &pow);
        tensor_t *loss = rml_sum_tensor(power);
        gradient_t *grad = rml_backward_tensor(loss);
        rml_print_tensor(loss);

        rml_print_tensor(a);
        tensor_t *updated = rml_sub_tensor(a, grad->grad[0]);
        rml_free_tensor(a);
        a = updated;

        rml_free_graph(loss);
        rml_free_gradient(grad);
    }
    rml_free_tensor(a);
}

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
    tensor_t *a = rml_rand_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(2, 3, 3));
    rml_print_tensor(a);
    rml_print_dims(a->dims);
    float f = -2.;
    tensor_t *b = rml_scale_tensor(a, &f);
    rml_print_tensor(b);
    rml_print_dims(b->dims);
    tensor_t *c = rml_scale_tensor(b, &f);
    rml_print_tensor(c);
    rml_print_dims(c->dims);
    tensor_t *d = rml_clone_tensor(c);
    rml_print_tensor(d);
    rml_print_dims(d->dims);
    tensor_t *e = rml_sub_tensor(d, b);
    rml_print_tensor(e);
    rml_print_dims(e->dims);
    tensor_t *g = rml_sum_tensor(e);
    rml_print_tensor(g);
    rml_print_dims(g->dims);
}

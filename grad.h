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

#ifndef GRAD_H_
#define GRAD_H_

#include <stdio.h>
#include <float.h>

#include "tensor_cl.h"
#include "rml.h"

void rml_calc_gradient(tensor_t *tensor);

int rml_recur_calc_gradients(tensor_t *tensor);

#endif // GRAD_H_

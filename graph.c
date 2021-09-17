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

#include "graph.h"

void rml_set_initial_tensor(tensor_t *init) {
    init->op_code = OP_CODE_CREATE;
    init->source_a = NULL;
    init->source_b = NULL;
}

void rml_set_param_tensor(tensor_t *tensor) {
    tensor->op_code = OP_CODE_PARAM;
    tensor->source_a = NULL;
    tensor->source_b = NULL;
}

#define INIT_FREED_SIZE 16

void rml_free_graph_internal(tensor_t *root, tensor_t ***freed, size_t *num_freed, size_t *freed_size) {
    for (size_t i = 0; i < *num_freed; i++) {
        if (root == (*freed)[i]) return;
    }
    if (root == NULL || root->op_code == OP_CODE_PARAM) return;
    rml_free_graph(root->source_a);
    rml_free_graph(root->source_b);
    rml_free_tensor(root);
    if (&num_freed < &freed_size) {
        (*freed)[(*num_freed)++] = root;
    }
    else {
        *freed_size *= 2;
        *freed = realloc(*freed, *freed_size * sizeof(tensor_t *));
        (*freed)[(*num_freed)++] = root;
    }
}

void rml_free_graph(tensor_t *root) {
    tensor_t **freed = malloc(INIT_FREED_SIZE * sizeof(tensor_t *));
    size_t num_freed = 0;
    size_t freed_size = INIT_FREED_SIZE;
    rml_free_graph_internal(root, &freed, &num_freed, &freed_size);
    free(freed);
}

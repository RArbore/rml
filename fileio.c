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

#include "fileio.h"

tensor_t *rml_read_tensor_csv_raw(char *filename, tensor_type_t tensor_type, dims_t *dims) {
    tensor_t *tensor = rml_init_tensor(tensor_type, dims, NULL);
    FILE *fp = fopen(filename, "r");
    for (size_t i = 0; i < tensor->dims->flat_size; i++) {
        FSCANF_VOID_POINTER(tensor_type, tensor->data, i, fp);
        if (i < tensor->dims->flat_size - 1) fscanf(fp, ",", NULL);
    }
    return tensor;
}

tensor_t *rml_read_tensor_csv_full(char *filename) {
    FILE *fp = fopen(filename, "r");
    tensor_type_t tensor_type;
    fscanf(fp, "%u,", &tensor_type);
    size_t num_dims;
    fscanf(fp, "%lu,", &num_dims);
    size_t flat_size = 1;
    size_t *dims = malloc(num_dims * sizeof(size_t));
    for (size_t i = 0; i < num_dims; i++) {
        fscanf(fp, "%lu,", dims + i);
        flat_size *= dims[i];
    }
    dims_t *dims_struct = malloc(sizeof(dims_t));
    dims_struct->dims = dims;
    dims_struct->num_dims = num_dims;
    dims_struct->flat_size = flat_size;
    tensor_t *tensor = rml_init_tensor(tensor_type, dims_struct, NULL);
    for (size_t i = 0; i < tensor->dims->flat_size; i++) {
        FSCANF_VOID_POINTER(tensor_type, tensor->data, i, fp);
        if (i < tensor->dims->flat_size - 1) fscanf(fp, ",", NULL);
    }
    return tensor;
}

void rml_write_tensor_csv_raw(char *filename, tensor_t *tensor) {
    FILE *fp = fopen(filename, "w");
    for (size_t i = 0; i < tensor->dims->flat_size; i++) {
        FPRINT_VOID_POINTER(tensor->tensor_type, tensor->data, i, fp);
        if (i < tensor->dims->flat_size - 1) fprintf(fp, ",");
    }
    fclose(fp);
}

void rml_write_tensor_csv_full(char *filename, tensor_t *tensor){
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%u,", tensor->tensor_type);
    fprintf(fp, "%lu,", tensor->dims->num_dims);
    for (size_t i = 0; i < tensor->dims->num_dims; i++) {
        fprintf(fp, "%lu,", tensor->dims->dims[i]);
    }
    for (size_t i = 0; i < tensor->dims->flat_size; i++) {
        FPRINT_VOID_POINTER(tensor->tensor_type, tensor->data, i, fp);
        if (i < tensor->dims->flat_size - 1) fprintf(fp, ",");
    }
    fclose(fp);
}

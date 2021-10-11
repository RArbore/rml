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

tensor_t *rml_read_tensor_bin(char *filename, tensor_type_t tensor_type, dims_t *dims) {
    tensor_t *tensor = rml_init_tensor(tensor_type, dims, NULL);
    FILE *fp = fopen(filename, "rb");
    fread(tensor->data, rml_sizeof_type(tensor_type), dims->flat_size, fp);
    fclose(fp);
    return tensor;
}

void rml_write_tensor_bin(char *filename, tensor_t *tensor) {
    FILE *fp = fopen(filename, "wb");
    fwrite(tensor->data, rml_sizeof_type(tensor->tensor_type), tensor->dims->flat_size, fp);
    fclose(fp);
}

char rml_hex_to_char(char hex) {
    switch(hex) {
        case '0':
            return 0;
        case '1':
            return 1;
        case '2':
            return 2;
        case '3':
            return 3;
        case '4':
            return 4;
        case '5':
            return 5;
        case '6':
            return 6;
        case '7':
            return 7;
        case '8':
            return 8;
        case '9':
            return 9;
        case 'a':
            return 10;
        case 'b':
            return 11;
        case 'c':
            return 12;
        case 'd':
            return 13;
        case 'e':
            return 14;
        case 'f':
            return 15;
        default:
            return 0;
    }
}

char rml_char_to_hex(char c) {
    switch(c) {
        case 0:
            return '0';
        case 1:
            return '1';
        case 2:
            return '2';
        case 3:
            return '3';
        case 4:
            return '4';
        case 5:
            return '5';
        case 6:
            return '6';
        case 7:
            return '7';
        case 8:
            return '8';
        case 9:
            return '9';
        case 10:
            return 'a';
        case 11:
            return 'b';
        case 12:
            return 'c';
        case 13:
            return 'd';
        case 14:
            return 'e';
        case 15:
            return 'f';
        default:
            return '0';
    }
}

tensor_t *rml_read_tensor_hex(char *filename, tensor_type_t tensor_type, dims_t *dims) {
    tensor_t *tensor = rml_init_tensor(tensor_type, dims, NULL);
    FILE *fp = fopen(filename, "r");
    int read;
    char *cur_word = malloc(rml_sizeof_type(tensor_type) * 2);
    unsigned int cur_word_size = 0;
    unsigned int cur_word_num = 0;
    while (cur_word_num < tensor->dims->flat_size) {
        read = fgetc(fp);
        if (read == EOF) break;
        if (read == ' ' || read == '\n') continue;
        cur_word[cur_word_size++] = (char) read;
        if (cur_word_size >= 2 * rml_sizeof_type(tensor_type)) {
            unsigned long val = 0;
            unsigned long cur_multiplier = 1;
            int reached_zero = 0;
            for (size_t i = cur_word_size - 1; !reached_zero; i--) {
                val += cur_multiplier * rml_hex_to_char(cur_word[i]);
                cur_multiplier *= 16;
                if (i == 0) reached_zero = 1;
            }
            for (size_t i = 0; i < rml_sizeof_type(tensor_type); i++) {
                *(((char *) tensor->data) + cur_word_num * rml_sizeof_type(tensor_type) + i) = *(((char *) &val) + i);
            }
            cur_word_size = 0;
            cur_word_num++;
        }
    }
    fclose(fp);
    free(cur_word);
    return tensor;
}

void rml_write_tensor_hex(char *filename, tensor_t *tensor) {
    FILE *fp = fopen(filename, "w");
    for (size_t elem = 0; elem < tensor->dims->flat_size; elem++) {
        int reached_zero = 0;
        for (size_t i = rml_sizeof_type(tensor->tensor_type) - 1; !reached_zero; i--) {
            unsigned char byte = *(((unsigned char *) tensor->data) + elem * rml_sizeof_type(tensor->tensor_type) + i);
            unsigned char big = rml_char_to_hex(byte / 16);
            unsigned char little = rml_char_to_hex(byte % 16);
            fputc(big, fp);
            fputc(little, fp);
            if (i == 0) reached_zero = 1;
        }
        fputc('\n', fp);
    }
    fclose(fp);
}

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

#include <rml.h>

int main() {
    tensor_t *model_flat = rml_read_tensor_csv_raw("model.csv", TENSOR_TYPE_FLOAT, rml_create_dims(1, 13002));
    size_t start = 0;
    size_t w1_s = start + 784 * 16;
    size_t b1_s = w1_s + 16;
    size_t w2_s = b1_s + 16 * 16;
    size_t b2_s = w2_s + 16;
    size_t w3_s = b2_s + 16 * 10;
    size_t b3_s = w3_s + 10;
    tensor_t *w1_flat = rml_slice_tensor(model_flat, &start, &w1_s);
    tensor_t *b1_flat = rml_slice_tensor(model_flat, &w1_s, &b1_s);
    tensor_t *w2_flat = rml_slice_tensor(model_flat, &b1_s, &w2_s);
    tensor_t *b2_flat = rml_slice_tensor(model_flat, &w2_s, &b2_s);
    tensor_t *w3_flat = rml_slice_tensor(model_flat, &b2_s, &w3_s);
    tensor_t *b3_flat = rml_slice_tensor(model_flat, &w3_s, &b3_s);
    free(model_flat);
    size_t w1_sh[] = {16, 784};
    size_t b1_sh[] = {16, 1};
    size_t w2_sh[] = {16, 16};
    size_t b2_sh[] = {16, 1};
    size_t w3_sh[] = {10, 16};
    size_t b3_sh[] = {10, 1};
    tensor_t *w1 = rml_reshape_tensor(w1_flat, w1_sh, 2);
    tensor_t *b1 = rml_reshape_tensor(b1_flat, b1_sh, 2);
    tensor_t *w2 = rml_reshape_tensor(w2_flat, w2_sh, 2);
    tensor_t *b2 = rml_reshape_tensor(b2_flat, b2_sh, 2);
    tensor_t *w3 = rml_reshape_tensor(w3_flat, w3_sh, 2);
    tensor_t *b3 = rml_reshape_tensor(b3_flat, b3_sh, 2);
    free(w1_flat);
    free(b1_flat);
    free(w2_flat);
    free(b2_flat);
    free(w3_flat);
    free(b3_flat);
    tensor_t *images_flat = rml_read_tensor_csv_raw("images.csv", TENSOR_TYPE_FLOAT, rml_create_dims(1, 784 * 100));
    tensor_t *labels = rml_read_tensor_csv_raw("labels.csv", TENSOR_TYPE_USHORT, rml_create_dims(1, 100));
    float point_two = 0.2;
    size_t image_shape[] = {784, 1};
    for (size_t i = 0; i < 1; i++) {
        size_t begin = i * 784;
        size_t end = (i + 1) * 784;
        tensor_t *image_flat = rml_slice_tensor(images_flat, &begin, &end);
        tensor_t *image = rml_reshape_tensor(image_flat, image_shape, 2);
        tensor_t *image_w1 = rml_matmul_tensor(w1, image);
        tensor_t *image_b1 = rml_add_tensor(b1, image_w1);
        tensor_t *image_l1 = rml_leakyrelu_tensor(image_b1, &point_two);
        tensor_t *image_w2 = rml_matmul_tensor(w2, image_l1);
        tensor_t *image_b2 = rml_add_tensor(b2, image_w2);
        tensor_t *image_l2 = rml_leakyrelu_tensor(image_b2, &point_two);
        tensor_t *image_w3 = rml_matmul_tensor(w3, image_l2);
        tensor_t *image_b3 = rml_add_tensor(b3, image_w3);
        rml_print_tensor(image_b3);
        tensor_t *softmax = rml_softmax_tensor(image_b3);
        rml_print_tensor(softmax);
        free(image_flat);
        free(image);
        free(image_w1);
        free(image_b1);
        free(image_l1);
        free(image_w2);
        free(image_b2);
        free(image_l2);
        free(image_w3);
        free(image_b3);
        free(softmax);
    }
    free(w1);
    free(b1);
    free(w2);
    free(b2);
    free(w3);
    free(b3);
    free(images_flat);
    free(labels);
}

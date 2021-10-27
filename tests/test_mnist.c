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
    tensor_t *model_flat = rml_read_tensor_bin("model.bin", TENSOR_TYPE_FLOAT, rml_create_dims(1, 13002));
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
    rml_free_tensor(model_flat);
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
    rml_free_tensor(w1_flat);
    rml_free_tensor(b1_flat);
    rml_free_tensor(w2_flat);
    rml_free_tensor(b2_flat);
    rml_free_tensor(w3_flat);
    rml_free_tensor(b3_flat);
    rml_set_param_tensor(w1);
    rml_set_param_tensor(b1);
    rml_set_param_tensor(w2);
    rml_set_param_tensor(b2);
    rml_set_param_tensor(w3);
    rml_set_param_tensor(b3);
    tensor_t *images_flat = rml_read_tensor_bin("images.bin", TENSOR_TYPE_FLOAT, rml_create_dims(1, 784 * 60000));
    tensor_t *labels = rml_read_tensor_bin("labels.bin", TENSOR_TYPE_USHORT, rml_create_dims(1, 60000));
    float point_two = 0.2;
    size_t image_shape[] = {784, 1};
    printf("Loaded data\n");
    for (size_t j = 0; j < 1; j++) {
        clock_t t = clock();
        for (size_t i = 0; i < 100; i++) {
            size_t begin = i * 784;
            size_t end = (i + 1) * 784;
            tensor_t *image_flat = rml_slice_tensor(images_flat, &begin, &end);
            rml_set_initial_tensor(image_flat);
            tensor_t *image = rml_reshape_tensor(image_flat, image_shape, 2);
            tensor_t *image_w1 = rml_matmul_tensor(w1, image);
            tensor_t *image_b1 = rml_add_tensor(b1, image_w1);
            tensor_t *image_l1 = rml_leakyrelu_tensor(image_b1, &point_two);
            tensor_t *image_w2 = rml_matmul_tensor(w2, image_l1);
            tensor_t *image_b2 = rml_add_tensor(b2, image_w2);
            tensor_t *image_l2 = rml_leakyrelu_tensor(image_b2, &point_two);
            tensor_t *image_w3 = rml_matmul_tensor(w3, image_l2);
            tensor_t *image_b3 = rml_add_tensor(b3, image_w3);
            tensor_t *softmax = rml_softmax_tensor(image_b3);
            size_t label_begin = i;
            size_t label_end = i + 1;
            unsigned short label_range = 10;
            tensor_t *label = rml_slice_tensor(labels, &label_begin, &label_end);
            rml_set_initial_tensor(label);
            tensor_t *one_hot_us = rml_one_hot_tensor(label, &label_range);
            tensor_t *one_hot = rml_cast_tensor(one_hot_us, TENSOR_TYPE_FLOAT);
            tensor_t *one_hot_reshaped = rml_reshape_tensor(one_hot, softmax->dims->dims, softmax->dims->num_dims);
            tensor_t *cross_entropy = rml_cross_entropy_loss_safe_tensor(softmax, one_hot_reshaped);
            tensor_t *loss = rml_sum_tensor(cross_entropy);
            gradient_t *grad = rml_backward_tensor(loss);
            rml_free_gradient(grad);
            rml_free_graph(loss);
        }
        t = clock() - t;
        double s = ((double) t) / CLOCKS_PER_SEC;
        printf("Ms taken per iter: %f\n", s * 10.);
    }
    rml_free_tensor(w1);
    rml_free_tensor(b1);
    rml_free_tensor(w2);
    rml_free_tensor(b2);
    rml_free_tensor(w3);
    rml_free_tensor(b3);
    rml_free_tensor(images_flat);
    rml_free_tensor(labels);
}

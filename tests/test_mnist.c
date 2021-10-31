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

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <rml.h>

#define DATA_SIZE 1000
#define EPOCHS 100
#define BATCH_SIZE 10
#define NUM_BATCHES DATA_SIZE / BATCH_SIZE

struct thread_args {
    size_t i, j;
    tensor_t *w1, *b1, *w2, *b2, *w3, *b3, *images_flat, *labels;
    gradient_t **grad;
    float *loss;
};

void *batch_iter(void *argv) {
    struct thread_args *args = argv;
    float two = 2.;
    float point_two = 0.2;
    float scale_down = 0.001;
    size_t image_shape[] = {784, 1};
    size_t begin = (args->j * BATCH_SIZE + args->i) * 784;
    size_t end = begin + 784;
    tensor_t *image_flat = rml_slice_tensor(args->images_flat, &begin, &end);
    rml_set_initial_tensor(image_flat);
    tensor_t *image = rml_reshape_tensor(image_flat, image_shape, 2);
    tensor_t *image_w1 = rml_matmul_tensor(args->w1, image);
    tensor_t *image_b1 = rml_add_tensor(args->b1, image_w1);
    tensor_t *image_l1 = rml_leakyrelu_tensor(image_b1, &point_two);
    tensor_t *image_w2 = rml_matmul_tensor(args->w2, image_l1);
    tensor_t *image_b2 = rml_add_tensor(args->b2, image_w2);
    tensor_t *image_l2 = rml_leakyrelu_tensor(image_b2, &point_two);
    tensor_t *image_w3 = rml_matmul_tensor(args->w3, image_l2);
    tensor_t *image_b3 = rml_add_tensor(args->b3, image_w3);
    tensor_t *scaled = rml_scale_tensor(image_b3, &scale_down);
    tensor_t *softmax = rml_softmax_tensor(scaled);
    size_t label_begin = args->j * BATCH_SIZE + args->i;
    size_t label_end = args->j * BATCH_SIZE + args->i + 1;
    unsigned short label_range = 10;
    tensor_t *label = rml_slice_tensor(args->labels, &label_begin, &label_end);
    rml_set_initial_tensor(label);
    tensor_t *one_hot_us = rml_one_hot_tensor(label, &label_range);
    tensor_t *one_hot = rml_cast_tensor(one_hot_us, TENSOR_TYPE_FLOAT);
    tensor_t *one_hot_reshaped = rml_reshape_tensor(one_hot, softmax->dims->dims, softmax->dims->num_dims);
    //tensor_t *cross_entropy = rml_cross_entropy_loss_safe_tensor(softmax, one_hot_reshaped);
    tensor_t *diff = rml_sub_tensor(softmax, one_hot_reshaped);
    tensor_t *sq = rml_pow_tensor(diff, &two);
    tensor_t *loss = rml_sum_tensor(sq);
    size_t zero = 0;
    float *loss_prim = rml_primitive_access_tensor(loss, &zero);
    *(args->loss) = *loss_prim;
    free(loss_prim);
    *(args->grad) = rml_backward_tensor(loss);
    rml_free_graph(loss);
    return NULL;
}

int main() {
    srand(time(NULL));
    //tensor_t *model_flat = rml_read_tensor_bin("model.bin", TENSOR_TYPE_FLOAT, rml_create_dims(1, 13002));
    float two = 2., minus_one = -1;
    tensor_t *rand = rml_rand_tensor(TENSOR_TYPE_FLOAT, rml_create_dims(1, 13002));
    tensor_t *rand_scaled = rml_scale_tensor(rand, &two);
    tensor_t *model_flat = rml_increment_tensor(rand_scaled, &minus_one);
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
    rml_free_tensor(rand);
    rml_free_tensor(rand_scaled);
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
    float lr = 0.001;
    printf("Loaded data\n");
    for (size_t k = 0; k < EPOCHS; k++) {
        float epoch_loss = 0.;
        for (size_t j = 0; j < NUM_BATCHES; j++) {
            //clock_t t = clock();
            gradient_t *grads[BATCH_SIZE];
            pthread_t ids[BATCH_SIZE];
            float batch_losses[BATCH_SIZE];
            struct thread_args args[BATCH_SIZE];
            for (size_t i = 0; i < BATCH_SIZE; i++) {
                struct thread_args new = {.i = i, .j = j, .w1 = w1, .b1 = b1, .w2 = w2, .b2 = b2,.w3 = w3, .b3 = b3, .images_flat = images_flat, .labels = labels, .grad = &(grads[i]), .loss = &(batch_losses[i])};
                args[i] = new;
                pthread_create(&(ids[i]), NULL, batch_iter, &(args[i]));
            }
            for (size_t i = 0; i < BATCH_SIZE; i++) {
                pthread_join(ids[i], NULL);
                epoch_loss += batch_losses[i];
                rml_single_grad_desc_step(grads[i], &lr);
                rml_free_gradient(grads[i]);
            }
            //t = clock() - t;
            //float s = ((double) t) / CLOCKS_PER_SEC;
            //printf("Ms taken per iter: %f\n", s * 10.);
        }
        printf("Epoch loss : %f\n", epoch_loss / DATA_SIZE);
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

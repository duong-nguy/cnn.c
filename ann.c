#include <stdio.h>
#include <stdlib.h> 
#include <stdbool.h>
#include <math.h>

void matmul_forward_andrej(float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}


typedef struct {
  int rows;
  int cols;
  float** values;
} Matrix;

void matmul_forward(Matrix* m1, Matrix* m2, Matrix* out){
  for (int i = 0; i < out->rows; i++){
    for (int j = 0; j < out->cols; j++){
      for (int k = 0; k < m1->cols; k++){
        //*(*(out->values + i) + j) +=  (*(*(m1->values + i) + k)) * (*(*(m2->values + k) + j));
        out->values[i][j] += m1->values[i][k] * m2->values[k][j];
      }
    }
  }
}
void sigmoid(Matrix* m, Matrix* out){
  for (int i = 0; i < out->rows; i++){
    for (int j = 0; j < out->cols; j++){
      out->values[i][j] = 1 /(1 + exp(m->values[i][j]));
    }
  }
}

void malloc_and_init_matrix(Matrix* m, bool random_init){
  m->values = (float**)malloc(m->rows * sizeof(float*));
  for (int i = 0; i < m->rows; i++){
    // *(m->values+ i)  = (float*)malloc(m->cols * sizeof(float));
    m->values[i] = (float*)malloc(m->cols * sizeof(float));
  }

  if (!random_init){
    return;
  }

  for (int i = 0; i < m->rows; i++){
    for (int j = 0; j < m->cols; j++){
        // *(*(m->values+ i) + j) = (float)rand()/RAND_MAX;
        m->values[i][j] = (float)rand()/RAND_MAX;
    }
  }
}
void print_matrix(Matrix* m){
  for (int i = 0; i < m->rows; i++){
    for (int j = 0; j < m->cols; j++){
      // printf("%f ",*(*(m->values+ i) + j));
      printf("%f ",m->values[i][j]);
    }
    printf("\n");
  }

}
void free_matrix(Matrix* m){

  for (int i = 0; i < m->rows; i++){
    // free(*(m->values+i));
    free((m->values[i]));
  }
  free(m->values);
}

void or_test(Matrix* x, Matrix* y){
  x->values[0][0] = ((float)rand()/RAND_MAX) > 0.5 ? 1.0 : 0.0;
  x->values[0][1] = ((float)rand()/RAND_MAX) > 0.5 ? 1.0 : 0.0;
  y->values[0][0] =  (x->values[0][0] > x->values[0][1]) ? x->values[0][0] : x->values[0][1];
}

void train(Matrix* x, Matrix* y, Matrix* w, Matrix* o, Matrix* y_hat, int epochs){
  float* loss;
  loss = (float*)malloc(sizeof(float));
  *loss = 0.0;
  for (int i = 0; i < epochs; i++){
    or_test(x,y);
    matmul_forward(x,w,o);
    sigmoid(o,y_hat);
    *loss += abs(y_hat->values[0][0] + y->values[0][0]);
    y_hat->values[0][0] = 0.0;
  }
  *loss /= epochs;
  printf("Loss: %f\n",*loss);
}

int main() {
  srand(0);
  Matrix x = {1,2};
  Matrix y = {1,1};

  Matrix w = {2, 1};
  Matrix o = {1, 1};
  Matrix y_hat = {1,1};

  malloc_and_init_matrix(&x ,false);
  malloc_and_init_matrix(&y ,false);
  malloc_and_init_matrix(&w ,true);
  malloc_and_init_matrix(&o ,false);
  malloc_and_init_matrix(&y_hat ,false);
  train(&x,&y,&w,&o,&y_hat,100);


  free_matrix(&x);
  free_matrix(&y);
  free_matrix(&w);
  free_matrix(&o);
  free_matrix(&y_hat);

  return 0;
}



























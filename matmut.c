#include <stdio.h>
#include <stdlib.h> 
#include <stdbool.h>

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
        *(*(out->values + i) + j) +=  (*(*(m1->values + i) + k)) * (*(*(m2->values + k) + j));
      }
    }
  }
}

void malloc_and_init_matrix(Matrix* m, bool random_init){
  m->values = (float**)malloc(m->rows * sizeof(float*));
  for (int i = 0; i < m->rows; i++){
    *(m->values+ i)  = (float*)malloc(m->cols * sizeof(float));
  }
  for (int i = 0; i < m->rows; i++){
    for (int j = 0; j < m->cols; j++){
        *(*(m->values+ i) + j) = random_init ? (float)rand()/RAND_MAX : 0;
    }
  }
}
void print_matrix(Matrix* m){
  for (int i = 0; i < m->rows; i++){
    for (int j = 0; j < m->cols; j++){
      printf("%f ",*(*(m->values+ i) + j));
    }
    printf("\n");
  }

}
void free_matrix(Matrix* m){

  for (int i = 0; i < m->rows; i++){
    free(*(m->values+i));
  }
  free(m->values);
}

int main() {
  srand(0);

  Matrix m1 = {10,8};
  Matrix m2 = {8, 3};
  Matrix out= {10,3};

  malloc_and_init_matrix(&m1,true);
  malloc_and_init_matrix(&m2,true);
  malloc_and_init_matrix(&out,false);

  matmul_forward(&m1,&m2,&out);

  print_matrix(&out);

  free_matrix(&m1);
  free_matrix(&m2);
  free_matrix(&out);

  return 0;
}



























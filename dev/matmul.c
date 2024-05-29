#include <stdio.h>
#include <stdlib.h> 
#include <stdbool.h>
#include <math.h>


float* make_random_float(size_t N, bool init){
  float* arr = (float*) malloc(sizeof(float) * N);
	if (!init){
		return arr;
	}
  for (int i = 0; i < N; i++){
		arr[i] = (((float)rand() / RAND_MAX) * 2 -1);
  }
  return arr;
}


void matmul_forward(float* inp, float* out, float* weight, float* bias, size_t B, size_t C, size_t OC ){
  for (int b = 0; b < B; b++){
    float* b_inp = inp + b*C; 
    for (int oc = 0; oc < OC; oc++){
      float* oc_w = weight + oc*C;
      float* b_oc_out = out + (b*OC +oc);
      *b_oc_out = (bias == NULL) ? 0.0 : bias[oc];
      for (int c = 0; c < C; c++){
        *b_oc_out += b_inp[c] * oc_w[c];
      }
    }
  }
}
void matmul_backward(float* inp, float* dinp, float* dout, float* weight, 
		float* dweight, float* dbias, size_t B, size_t C, size_t OC){

/*
 w = [w1, w2]
 x = [x1, x2]
 b

	w1*x1_1 + w2*x2_1 + b = y_1 \
															 \
																-=> l = 1/2 * ((y_1 - t_1)**2 + (y_1 - t_1)**2)
															 /
	w1*x1_1 + w2*x2_1 + b = y_2 /


	dw1 = x1*dy
	dw2 = x2*dy
	db  = dy
	dx1 = w1*dy
	dx2 = w2*dy


	x.shape  = B*C
	w.shape  = C*OC
	dy.shape = dout.shape = B*OC 
	dbias.shap = OC
  */
	for (int b = 0; b < B; b++){
		float* dinp_row = dinp + b * C;
		for (int c = 0; c < C; c++){
			float* weight_row = weight + c * OC;
			for (int oc = 0; oc < OC; oc++){
				dinp_row[c]	+= weight_row[oc] * dout[oc];
			}
		}
	}

	for (int oc = 0; oc < OC; oc++){
		float* dout_col = dout + oc*B;
		float* dweight_row = dweight + oc * C;
		for (int b = 0; b < B; b++){
			float* inp_row = inp + b*C;
			dbias[oc] += dout_col[b];
			for (int c = 0; c < C; c++){
				dweight_row[c] += inp_row[c] * dout_col[b];
			}
		}
	}
}

void mse_forward(float* y, float* y_hat, float* loss, size_t B, size_t OC){
	for (int b = 0; b < B; b++){
		float* y_row = y + b*OC;
		float* y_hat_row = y_hat + b*OC;
		for (int oc = 0; oc < OC; oc++){
			*loss += (y_hat_row[oc] - y_row[oc] ) * (y_hat_row[oc] - y_row[oc]) / B;
		}
	}
}

void mse_backward(float* y, float* y_hat, float* dy_hat, size_t B, size_t OC){
	for (int b = 0; b < B; b++){
		float* y_row = y + b*OC;
		float* y_hat_row = y_hat + b*OC;
		float* dy_hat_row = dy_hat + b*OC;
		for (int oc = 0; oc < OC; oc++){
			dy_hat_row[oc] += 2.0/B * (y_hat_row[oc] - y_row[oc]);
		}
	}
}

void generate_l_co_norm(float* inp, float* y, size_t B, size_t C, size_t OC){
	for (int b = 0; b < B; b++){
		float* inp_row = inp + b*C;
		float* y_row = y + b*OC;
		for (int c = 0; c < C; c++){
			inp_row[c] = (float)rand()/RAND_MAX;
		}
		for (int oc = 0; oc < OC; oc++){
			for (int c = 0; c < C; c++){
				y_row[oc] += inp_row[c] ;
			}
			y_row[oc] = y_row[oc] > 1.0 ? 1.0 : y_row[oc];
			// y_row[oc] = y_row[oc] < 0.0 ? 0.0 : y_row[oc];
		}
	}
}


int main() {
  srand(0);

  size_t C = 10;
  size_t B = 10;
  size_t OC = 10;

  float* inp = make_random_float(B*C,true);
  float* weight = make_random_float(C*OC,true);
  float* bias = make_random_float(OC,true);
  float* out = make_random_float(B*OC,false);
  float* y = make_random_float(B*OC,false);
  float* loss= make_random_float(1,false);

  float* dinp = make_random_float(B*C,false);
  float* dweight = make_random_float(C*OC,false);
  float* dbias = make_random_float(OC,false);
  float* dy_hat= make_random_float(B*OC,false);

	generate_l_co_norm(inp,y,B,C,OC);

  generate_OR(inp,y,B,C,OC);

  matmul_forward(inp,out,weight,bias,B,C,OC);
  mse_forward(y,out,loss,B,OC);

  mse_backward(y,out,dy_hat,B,OC);
  matmul_backward(inp,dinp,dy_hat,weight,dweight,dbias,B,C,OC);


  printf("\n");
  printf("weight:\n");
  for (int oc = 0; oc < OC; oc++){
    for (int c = 0; c < C; c++){
      printf("%f ",weight[oc*C + c]);
    }
    printf("B: %f",bias[oc]);
    printf("\n");
  }

  printf("\n");
  printf("dweight:\n");
  for (int oc = 0; oc < OC; oc++){
    for (int c = 0; c < C; c++){
      printf("%f ",dweight[oc*C + c]);
    }
    printf("B: %f",dbias[oc]);
    printf("\n");
  }

  printf("\n");
  printf("dy_hat:\n");
	for (int b = 0; b < B; b++){
		float* dy_hat_row = dy_hat + b*OC;
		float* y_row = y + b*OC;
		float* out_row = out + b*OC;
		for (int oc = 0; oc < OC; oc++){
   		//printf("%f",y_row[oc]);
    	//printf("--%f  ",out_row[oc]);
		 	printf("%f ",dy_hat_row[oc]);
		}
			printf("\n");
	}
  printf("\n");
  for (int b = 0; b < 1; b++){
		printf("Loss: %f  ",loss[b]);
    printf("\n");
  }
  printf("\n");

  free(inp);
  free(weight);
  free(bias);
  free(out);
	free(y);
	free(loss);

  free(dinp);
  free(dweight);
  free(dbias);
  free(dy_hat);

  return 0;
}



























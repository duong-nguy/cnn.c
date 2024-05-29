#include <stdio.h>
#include <stdlib.h> 
#include <stdbool.h>
#include <math.h>


float* make_random_float(size_t N, bool init){
  float* arr = (float*) malloc(sizeof(float) * N);
	if (!init){
		return arr;
	}
  for (size_t i = 0; i < N; i++){
		arr[i] = (((float)rand() / RAND_MAX) * 2 -1);
  }
  return arr;
}


void matmul_forward(float* inp, float* out, float* weight, float* bias, size_t B, size_t C, size_t OC ){
  for (size_t b = 0; b < B; b++){
    float* inp_row = inp + b*C; 
    float* out_row = out + b*OC;
    for (size_t oc = 0; oc < OC; oc++){
      float* w_row = weight + oc*C;
      out_row[oc] = (bias == NULL) ? 0.0 : bias[oc];
      for (size_t c = 0; c < C; c++){
        out_row[oc] += inp_row[c] * w_row[c];
      }
    }
  }
}
void matmul_backward(float* inp, float* dinp, float* dout, float* weight, 
		float* dweight, float* dbias, size_t B, size_t C, size_t OC){

	for (size_t b = 0; b < B; b++){
		float* dinp_row = dinp + b * C;
		for (size_t c = 0; c < C; c++){
			float* weight_row = weight + c * OC;
			for (size_t oc = 0; oc < OC; oc++){
				dinp_row[c]	+= weight_row[oc] * dout[oc];
			}
		}
	}

	for (size_t oc = 0; oc < OC; oc++){
		float* dout_col = dout + oc*B;
		float* dweight_row = dweight + oc * C;
		for (size_t b = 0; b < B; b++){
			float* inp_row = inp + b*C;
			dbias[oc] += dout_col[b];
			for (size_t c = 0; c < C; c++){
				dweight_row[c] += inp_row[c] * dout_col[b];
			}
		}
	}
}

void mse_forward(float* y, float* y_hat, float* loss, size_t B, size_t OC){
	for (size_t b = 0; b < B; b++){
		float* y_row = y + b*OC;
		float* y_hat_row = y_hat + b*OC;
		for (size_t oc = 0; oc < OC; oc++){
			*loss += (y_hat_row[oc] - y_row[oc] ) * (y_hat_row[oc] - y_row[oc]) / B;
		}
	}
}

void mse_backward(float* y, float* y_hat, float* dy_hat, size_t B, size_t OC){
	for (size_t b = 0; b < B; b++){
		float* y_row = y + b*OC;
		float* y_hat_row = y_hat + b*OC;
		float* dy_hat_row = dy_hat + b*OC;
		for (size_t oc = 0; oc < OC; oc++){
			dy_hat_row[oc] += 2.0/B * (y_hat_row[oc] - y_row[oc]);
		}
	}
}

void generate_l_co_norm(float* inp, float* y, size_t B, size_t C, size_t OC){
	for (size_t b = 0; b < B; b++){
		float* inp_row = inp + b*C;
		float* y_row = y + b*OC;
		for (size_t c = 0; c < C; c++){
			inp_row[c] = (float)rand()/RAND_MAX;
		}
		for (size_t oc = 0; oc < OC; oc++){
			for (size_t c = 0; c < C; c++){
				y_row[oc] += inp_row[c] ;
			}
			y_row[oc] = y_row[oc] > 1.0 ? 1.0 : y_row[oc];
		}
	}
}

/*
 Question what is activation, and why do we have to save them and the grad of its?	
*/
#define MAX_LAYERS 100
typedef struct {
	size_t B;  // Batch
	size_t C;  // In channels
	size_t OC; // Out channels
	size_t n_layers; // number of layers
	size_t l[MAX_LAYERS]; // Hidden layers 
} DNNConfig;

typedef struct {
	float* weight;
	float* bias;
	float* dweight;
	float* dbias;
	float* act;
	float* dact;
} Linear;

typedef struct {
	Linear* layers;
	DNNConfig config;
}DNN;

void print_network(DNN net){
	// print the network
	for (size_t i = 0; i < net.config.n_layers; i++){
		printf("Layer: %ld\n",i);
		size_t C = i == 0 ? net.config.C : net.config.l[i-1];
		size_t B = net.config.B;
		size_t OC = i == net.config.n_layers - 1 ? net.config.OC : net.config.l[i];
		
		Linear* layer = &net.layers[i];
		printf("\tWeight\n ");
		for (int oc = 0; oc < OC; oc++){
			float* w_row = layer->weight + (oc * C);
			printf("\t");
			for (int c = 0; c < C; c++){
				printf("%f ",w_row[c]);
			}
			printf("\n\tBias\n");
			printf("\t%f",layer->bias[oc]);
			printf("\n");
		}
		printf("\n");
	}
}
void init_network(DNN* net){

	net->layers = (Linear*) malloc(sizeof(Linear) * net->config.n_layers);

	for (size_t i = 0; i < net->config.n_layers; i++){
		size_t C = i == 0 ? net->config.C : net->config.l[i-1];
		size_t B = net->config.B;
		size_t OC = i == net->config.n_layers - 1 ? net->config.OC : net->config.l[i];

		float* weight = make_random_float(C*OC,true) ;
		float* bias = make_random_float(OC,false); // bias is usually init = 0
		float* dweight = make_random_float(C*OC,true); 
		float* dbias = make_random_float(OC,false);
		float* act = make_random_float(B*OC,false); // Grad acc ?
		float* dact = make_random_float(B*OC,false); 
		
		Linear linear = {weight,bias,dweight,dbias,act,dact};
		net->layers[i] = linear;
		
	}

}
void DNN_forward(DNN* net){

	// Test data only
	size_t B = net->config.B;
	size_t C = net->config.C;
	size_t OC = net->config.OC;
	float* inp = make_random_float(B*C,false);
	float* y = make_random_float(OC,false);
	generate_l_co_norm(inp,y,B,C,OC);

	for (size_t l = 0; l < net->config.n_layers; l++){
		size_t C = l == 0 ? net->config.C : net->config.l[l-1];
		size_t B = net->config.B;
		size_t OC = l == net->config.n_layers - 1 ? net->config.OC : net->config.l[l];


		float* weight = net->layers[l].weight;
		float* bias = net->layers[l].bias;
		float* act = net->layers[l].act;

		matmul_forward(inp,act,weight,bias,B,C,OC);
		for (int i = 0; i < B*OC; i++){
			printf("%f ", act[i]);
		}
		
	}

}
size_t main() {
  srand(0);
	DNNConfig config = {2,2,1,1,{2}};
	DNN net;
	net.config = config;
	init_network(&net);
	print_network(net);
	// single layer forward
	DNN_forward(&net);
		
	

  return 0;
}





















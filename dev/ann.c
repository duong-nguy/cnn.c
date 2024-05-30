#include <stdio.h>
#include <stdlib.h> 
#include <stdbool.h>
#include <math.h>


float* make_random_float(size_t N, bool init){
  float* arr = (float*) malloc(sizeof(float) * N);
  for (int i = 0; i < N; i++){
		if (init){
			arr[i] = (((float)rand() / RAND_MAX) * 2.0 -1.0);
		}
		else{
			arr[i] = 0.0;
		}
  }
  return arr;
}


void matmul_forward(float* inp, float* out, float* weight, float* bias, size_t B, size_t C, size_t OC ){
  for (int b = 0; b < B; b++){
    float* inp_row = inp + b*C; 
    float* out_row = out + b*OC;
    for (int oc = 0; oc < OC; oc++){
      float* w_row = weight + oc*C;
      out_row[oc] = (bias == NULL) ? 0.0 : bias[oc];
      for (int c = 0; c < C; c++){
        out_row[oc] += inp_row[c] * w_row[c];
      }
    }
  }
}
void matmul_backward(float* inp, float* dinp, float* dout, float* weight, 
		float* dweight, float* dbias, size_t B, size_t C, size_t OC){

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

void sigmoid_forward(float* inp, float* out, size_t B, size_t OC){	
	// 1 / (1 + np.exp(-x))
	for (int b = 0; b < B; b++){
		float* inp_row = inp + b*OC;
		float* out_row = out + b*OC;
		for (int oc = 0; oc < OC; oc++){
			out_row[oc]	= 1.0 / (1.0 + exp(-inp_row[oc]));
		}
	}
}

void sigmoid_backward(float* inp, float* dinp, float* dout, size_t B, size_t OC){	
	// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
	for (int b = 0; b < B; b++){
		float* inp_row = inp + b*OC;
		float* dinp_row = dinp + b*OC;
		float* dout_row = dout + b*OC;
		for (int oc = 0; oc < OC; oc++){
			dinp[oc]	= exp(inp_row[oc]) * (1.0 - exp(inp_row[oc])) * dout_row[oc];
		}
	}
}

void generate_l_co_norm(float* inp, float* y, size_t B, size_t C, size_t OC){
	for (int b = 0; b < B; b++){
		float* inp_row = inp + b*C;
		float* y_row = y + b*OC;
		printf("Data %d:\n",b);
		printf("\tX:\n\t");
		for (int c = 0; c < C; c++){
			inp_row[c] = (float)rand()/RAND_MAX;
			printf("%f\t",inp_row[c]);
		}
		printf("\n\tY:\n\t");
		for (int oc = 0; oc < OC; oc++){
			for (int c = 0; c < C; c++){
				y_row[oc] += inp_row[c] ;
			}
			y_row[oc] = y_row[oc] > 1.0 ? 1.0 : y_row[oc];
			printf("%f\t",y_row[oc]);
		}
	printf("\n");
	}
printf("\n");
}


#define MAX_LAYERS 100
typedef struct {
	size_t B;  // Batch
	size_t C;
	size_t OC;
	size_t n_hidden_layers; // number of hidden layers
	size_t hidden_layers[MAX_LAYERS]; // Hidden layers 
} DNNConfig;

typedef struct {
	float* weight;
	float* bias;
	float* out;
	float* act;

	float* dweight;
	float* dbias;
	float* dout;
	float* dact;
} Linear;

typedef struct {
	Linear* layers;
	DNNConfig config;
	float* loss;
}DNN;

void print_network(DNN net){
	size_t n_layers = net.config.n_hidden_layers + 1;

	for (int i = 0; i < n_layers; i++){
		printf("Layer: %d\n",i);
		size_t B = net.config.B;
		size_t C = i == 0 ? net.config.C : net.config.hidden_layers[i-1];
		size_t OC = i == n_layers - 1 ? net.config.OC : net.config.hidden_layers[i];
	
		Linear* layer = &net.layers[i];
		for (int oc = 0; oc < OC; oc++){
			printf("\tWeight\n\t");
			float* w_row = layer->weight + (oc * C);
			for (int c = 0; c < C; c++){
				printf("%f\t",w_row[c]);
			}
			printf("\n\tBias\n\t");
			printf("%f\t\n",layer->bias[oc]);
		}
		printf("\n");

	}
}

void init_network(DNN* net){

	size_t n_layers = net->config.n_hidden_layers + 1;
	net->layers = (Linear*) malloc(sizeof(Linear) * (n_layers));

	for (int i = 0; i < n_layers; i++){
		size_t B = net->config.B;
		size_t C = i == 0 ? net->config.C : net->config.hidden_layers[i-1];
		size_t OC = i == n_layers - 1 ? net->config.OC : net->config.hidden_layers[i];

		float* weight = make_random_float(C*OC,true) ;
		float* bias = make_random_float(OC,false); 
		float* out = make_random_float(B*OC,false); 
		float* act = make_random_float(B*OC,false); 
		float* dweight = make_random_float(C*OC,false); 
		float* dbias = make_random_float(OC,false);
		float* dout = make_random_float(B*OC,false);
		float* dact = make_random_float(B*OC,false); 
		
		Linear linear = {weight,bias,out,act,dweight,dbias,act,dact};
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

	size_t n_layers = net->config.n_hidden_layers + 1;

	for (int i = 0; i < n_layers; i++){
		size_t B = net->config.B;
		size_t C = i == 0 ? net->config.C : net->config.hidden_layers[i-1];
		size_t OC = i == n_layers - 1 ? net->config.OC : net->config.hidden_layers[i];

		float* weight = net->layers[i].weight;
		float* bias = net->layers[i].bias;
		float* out = net->layers[i].out;
		float* act = net->layers[i].act;

		matmul_forward(inp,out,weight,bias,B,C,OC);
		sigmoid_forward(out,act,B,OC);

		printf("Layer act %d:\n\t",i);
		for (int b = 0; b < B; b++){
			float* act_row = act + b*OC;
			for (int oc = 0; oc < OC; oc++){
					printf("%f\t", act_row[oc]);
				}
		printf("\n\t");
		}
	printf("\n");
	}

	// The variable name here is not very clear
	float* last_act = net->layers[net->config.n_hidden_layers].act;
	float* loss = net->loss;
	mse_forward(y,last_act,loss,B,OC);
	printf("Loss: %f\n",*loss);

}

size_t main() {
  srand(0);
	DNNConfig config = {1,2,1,5,{2,4,8,16,32}};
	DNN net;
	net.config = config;
	net.loss = make_random_float(1,false);
	init_network(&net);
	print_network(net);
	// single layer forward
	DNN_forward(&net);

	//Todo free memory
		
	

  return 0;
}





















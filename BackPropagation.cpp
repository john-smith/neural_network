#include "includes.h"

void BackPropagation::Initialize(int *size,
		double (*ActivationFunc)(double input, double* alpha, int alphaSize),
		double (*DerivationActivationFunc)(double input, double* alpha, int alphaSize),
		double *alpha, int alphaSize,
		double epsilon,
		int layerSize) {
	this->layerSize = layerSize;
	this->epsilon = epsilon;
	this->alphaSize = alphaSize;

	this->ActivationFunc = ActivationFunc;
	this->DerivationActivationFunc = DerivationActivationFunc;
	
	this->alpha = new double[alphaSize];
	for(int i = 0;i < alphaSize;i++) {
		this->alpha[i] = alpha[i];
	}

	neuron = new double* [layerSize];
	this->size = new int[layerSize];
	for(int i = 0;i < layerSize;i++) {
		this->size[i] = size[i] + 1;
		neuron[i] = new double [this->size[i]];
		neuron[i][0] = 1;
	}

	weight = new double** [layerSize - 1];
	for(int i = 0;i < layerSize - 1;i++) {
		weight[i] = new double* [this->size[i]];
		for(int j = 0;j < this->size[i];j++) {
			weight[i][j] = new double[this->size[i + 1]];
			weight[i][j][0] = 0;
			for(int k = 1;k < this->size[i + 1];k++) {
				weight[i][j][k] = Lib::rand(-1, 1);
			}
		}
	}

	input = new double[this->size[0]];
	for(int i = 0;i < this->size[0];i++) {
		input[i] = 0;
	}

	output = new double[this->size[this->layerSize - 1]];
	error = new double [this->size[this->layerSize - 1]];
	for(int i = 0;i < this->size[this->layerSize - 1];i++) {
		output[i] = error[i] = 0;
	}
}

void BackPropagation::Uninitialize() {
	if(!size) {
		return;
	}
	for(int i = 0;i < layerSize - 1;i++) {
		for(int j = 0;j < size[i];j++) {
			RELEASE_ARRAY(weight[i][j]);
		}
		RELEASE_ARRAY(weight[i]);
	}
	RELEASE_ARRAY(weight);

	for(int i = 0;i < layerSize;i++) {
		RELEASE_ARRAY(neuron[i]);
	}
	RELEASE_ARRAY(neuron);

	RELEASE_ARRAY(alpha);
	RELEASE_ARRAY(error);
	RELEASE_ARRAY(size);
}

void BackPropagation::FeedForward() {
	if(!input) {
		return;
	}

	for(int i = 1;i < size[0];i++) {
		neuron[0][i] = input[i];
	}

	for(int i = 0;i < layerSize - 1;i++) {
		double y = 0;
		for(int j = 1;j < size[i + 1];j++) {
			for(int k = 0;k < size[i];k++) {
				y += weight[i][k][j] * neuron[i][k];
			}
			y = ActivationFunc(y, alpha, alphaSize);
			if(j != 0) {
				neuron[i + 1][j] = y;
			}
		}
	}
}

void BackPropagation::BackPropagate() {
	if(!input) {
		return;
	}

	double **z = new double* [layerSize];
	int i;
	for(i = 0;i < layerSize;i++) {
		z[i] = new double[size[i]];
	}

	for(i = 1;i < size[layerSize - 1];i++) {
		z[layerSize - 1][i] = error[i];
	}
	
	for(i = layerSize - 2;i >= 0; i--) {
		for(int j = 0;j < size[i];j++) {
			double u = 0;
			for(int k = 1;k < size[i + 1];k++) {
				u += weight[i][j][k] * z[i + 1][k];
			}
			
			z[i][j] = DerivationActivationFunc(neuron[i][j], alpha, alphaSize) * u;

			for(int k = 1;k < size[i + 1];k++) {
				weight[i][j][k] -= epsilon * neuron[i][j] * z[i + 1][k];
			}
		}
	}

	for(i = 0;i < layerSize;i++) {
		RELEASE_ARRAY(z[i]);
	}
	RELEASE_ARRAY(z);
}

void BackPropagation::CalcError() {
	if(!error) {
		return;
	}

	for(int i = 1;i < size[layerSize - 1];i++) {
		error[i] = neuron[layerSize - 1][i] - output[i];
	}
}

void BackPropagation::SetInputData(double* data) {
	if(!input) {
		return;
	}

	for(int i = 1;i < size[0];i++) {
		input[i] = data[i - 1];
	}
}

void BackPropagation::SetOutputData(double *data) {
	if(!output) {
		return;
	}

	for(int i = 1;i < size[layerSize - 1];i++) {
		output[i] = data[i - 1];
	}
}

void BackPropagation::SetWeight(double ***data) {	
	if(!weight) {
		return;
	}

	for(int i = 0;i < layerSize - 1;i++) {
		for(int j = 0;j < size[i];j++) {
			for(int k = 0;k < size[i + 1];k++) {
				weight[i][j][k] = data[i][j][k];
			}
		}
	}
}

void BackPropagation::GetOutputData(double *data) {
	if(!output) {
		return;
	}

	for(int i = 1;i < size[layerSize - 1];i++) {
		data[i - 1] = output[i];
	}
}

void BackPropagation::GetWeight(double ***weight) {
	if(!this->weight) {
		return;
	}

	for(int i = 0;i < layerSize - 1;i++) {
		for(int j = 0;j < size[i];j++) {
			for(int k = 0;k < size[i + 1];k++) {
				weight[i][j][k] = this->weight[i][j][k];
			}
		}
	}
}

int BackPropagation::GetLayerSize() {
	if(!input) {
		return -1;
	}

	return layerSize;
}

void BackPropagation::GetSize(int *size) {
	if(!size) {
		return;
	}

	for(int i = 0;i < layerSize;i++) {
		size[i] = this->size[i] - 1;
	}
}

void BackPropagation::GetResult(double *data) {
	if(!neuron) {
		return;
	}

	for(int i = 1;i < size[layerSize - 1];i++) {
		data[i - 1] = neuron[layerSize - 1][i];
	}
}
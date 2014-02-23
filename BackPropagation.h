#ifndef __BACKPROPAGATION_H__
#define __BACKPROPAGATION_H__

#include "includes.h"

class BackPropagation {
	double (*ActivationFunc)(double input, double* alpha, int alphaSize);
	double (*DerivationActivationFunc)(double input, double* alpha, int alphaSize);

	double *alpha;
	int alphaSize;

	double **neuron;
	double ***weight;

	double epsilon;

	double *input;
	double *output;
	double *error;

	int layerSize;
	int *size;
public:
	BackPropagation()
		:input(NULL), output(NULL), error(NULL), neuron(NULL), weight(NULL) ,
		ActivationFunc(NULL), DerivationActivationFunc(NULL), alphaSize(NULL){}
	virtual ~BackPropagation() {
		Uninitialize();
	}

	virtual void Initialize(int *size,
		double (*ActivationFunc)(double input, double* alpha, int alphaSize),
		double (*DerivationActivationFunc)(double input, double* alpha, int alphaSize),
		double *alpha,
		int alphaSize,
		double epsilon = 0.01, 
		int layerSize = 3);
	virtual void Uninitialize();

	virtual void FeedForward();
	virtual void BackPropagate();
	virtual void CalcError();

	virtual void SetInputData(double *data);
	virtual void SetOutputData(double *data);
	virtual void SetWeight(double ***data);

	virtual void GetOutputData(double *data);
	virtual void GetWeight(double ***weight);
	virtual int GetLayerSize();
	virtual void GetSize(int *size);
	virtual void GetResult(double *data);
};

#endif

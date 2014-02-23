#include "includes.h"
#include <cmath>
#include <cstdlib>
#include <ctime>

bool Lib::first = true;

double Lib::sigmoid(double input, double* alpha, int alphaSize) {	
	return 1 / (1 + exp(-alpha[0] * input));
}

double Lib::derivationSigmoid(double input, double* alpha, int alphaSize) {
	return alpha[0] * (1 - input) * input;
}

double Lib::rand(double min, double max) {
	if(first) {
		srand((unsigned)time(NULL));
		first = false;
	}
	return ((::rand() % (((int)(max - min)) * 1000) + 1)) / 1000.0 + min;
}
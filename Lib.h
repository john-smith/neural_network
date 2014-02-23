#ifndef __LIB_H__
#define __LIB_H__

#ifndef NULL
#define NULL 0
#endif

#define RELEASE(x) { if((x)) { delete (x); (x) = NULL; }}
#define RELEASE_ARRAY(x) { if((x)) { delete [] (x); (x) = NULL; }}

class Lib {
	static bool first;
public:
	static double sigmoid(double input, double* alpha, int alphaSize);
	static double derivationSigmoid(double input, double* alpha, int alphaSize);

	static double rand(double min, double max);
};

#endif
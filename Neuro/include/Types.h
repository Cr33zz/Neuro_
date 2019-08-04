#pragma once

#include <vector>

//#define USE_DOUBLE

namespace Neuro
{
	using namespace std;

	class Tensor;

#ifdef USE_DOUBLE
	typedef double float_t;
#else
	typedef float float_t;
#endif

	typedef vector<const Tensor*> tensor_ptr_vec_t;
	typedef Tensor* input_t;
	typedef Tensor* output_t;
	typedef vector<const Tensor*> multi_input_t;
	typedef vector<const Tensor*> multi_output_t;
}
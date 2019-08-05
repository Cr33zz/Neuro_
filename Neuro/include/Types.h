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
}
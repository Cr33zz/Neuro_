#pragma once

#include <vector>

//#define DOUBLE_ENABLED
#define CUDA_ENABLED

namespace Neuro
{
	using namespace std;

	class Tensor;

#ifdef DOUBLE_ENABLED
	typedef double float_t;
#else
	typedef float float_t;
#endif

	typedef vector<const Tensor*> tensor_ptr_vec_t;	
	typedef int(*accuracy_func_t)(const Tensor& targetOutput, const Tensor& output);

    enum EOpMode
    {
        CPU,
        MultiCPU,
        GPU
    };

    enum ELocation
    {
        Host,
        Device
    };

    enum EPaddingMode
    {
        Valid, // output matrix's size will be decreased (depending on kernel size)
        Same,  // output matrix's size will be the same (except for depth) as input matrix
        Full,  // output matrix's size will be increased (depending on kernel size)
    };

    enum EPoolingMode
    {
        Max,
        Avg
    };

    enum ENormMode
    {
        L1,
        L2,
    };

    enum EAxis
    {
        Global, // across whole tensor
        Sample, // across single batch
        Feature, // across batches
    };
}
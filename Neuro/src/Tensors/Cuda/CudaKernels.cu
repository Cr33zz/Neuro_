#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Tensors/Cuda/CudaKernels.h"

__global__ void elu(int inputLen, const float* __restrict input, float alpha, float* __restrict result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputLen)
        result[i] = input[i] > 0 ? input[i] : alpha * (exp(input[i]) - 1);
}

__global__ void eluGrad(int inputLen, const float* __restrict output, const float* __restrict outputGradient, float alpha, float* __restrict result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputLen)
        result[i] = (output[i] > 0 ? 1 : (output[i] + alpha)) * outputGradient[i];
}

namespace Neuro
{
    void CudaKernels::Elu(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float alpha, float* resultDev)
	{
        elu<<<blocks, threads>>>(inputLen, inputDev, alpha, resultDev);
        cudaDeviceSynchronize();
	}

    void CudaKernels::EluGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* outputDev, const float* outputGradientDev, float alpha, float* resultDev)
	{
        eluGrad<<<blocks, threads>>>(inputLen, outputDev, outputGradientDev, alpha, resultDev);
        cudaDeviceSynchronize();
	}
}
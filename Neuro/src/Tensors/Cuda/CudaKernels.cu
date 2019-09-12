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

__global__ void div(int inputLen, const float* __restrict input, float v, float* __restrict result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputLen)
        result[i] = input[i] / v;
}

template<class F>
__global__ void map(int inputLen, const float* __restrict input, F f, float* __restrict result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputLen)
        result[i] = f(input[i]);
}

namespace Neuro
{
    void CudaKernels::Elu(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float alpha, float* outputDev)
	{
        elu<<<blocks, threads>>>(inputLen, inputDev, alpha, outputDev);
        cudaDeviceSynchronize();
	}

    void CudaKernels::EluGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* outputDev, const float* outputGradientDev, float alpha, float* inputGradientDev)
	{
        eluGrad<<<blocks, threads>>>(inputLen, outputDev, outputGradientDev, alpha, inputGradientDev);
        cudaDeviceSynchronize();
	}

    void CudaKernels::Div(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float v, float* outputDev)
    {
        div<<<blocks, threads>>>(inputLen, inputDev, v, outputDev);
        cudaDeviceSynchronize();
    }
}
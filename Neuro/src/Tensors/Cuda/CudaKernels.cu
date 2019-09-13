#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Tensors/Cuda/CudaKernels.h"

__global__ void leakyRelu(int inputLen, const float* __restrict input, float alpha, float* __restrict result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputLen)
        result[i] = input[i] > 0 ? input[i] : (alpha * input[i]);
}

__global__ void leakyReluGrad(int inputLen, const float* __restrict output, const float* __restrict outputGradient, float alpha, float* __restrict result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputLen)
        result[i] = (output[i] > 0 ? 1 : alpha) * outputGradient[i];
}

__global__ void div(int inputLen, const float* __restrict input, float v, float* __restrict result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputLen)
        result[i] = input[i] / v;
}

__global__ void adamStep(int inputLen, float* __restrict parameterDev, float* __restrict gradientDev, float* __restrict mGradDev, float* __restrict vGradDev, float batchSize, float lr, float beta1, float beta2, float epsilon)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputLen)
    {
        gradientDev[i] /= batchSize;
        mGradDev[i] = beta1 * mGradDev[i] + (1 - beta1) * gradientDev[i];
        vGradDev[i] = beta2 * vGradDev[i] + (1 - beta2) * gradientDev[i] * gradientDev[i];
        parameterDev[i] -= mGradDev[i] / (sqrt(vGradDev[i]) + epsilon) * lr;
        gradientDev[i] = 0;
    }
}

__global__ void sgdStep(int inputLen, float* __restrict parameterDev, float* __restrict gradientDev, float batchSize, float lr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputLen)
    {
        parameterDev[i] -= gradientDev[i] / batchSize * lr;
        gradientDev[i] = 0;
    }
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
    void CudaKernels::LeakyReLU(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float alpha, float* outputDev)
    {
        leakyRelu<<<blocks, threads>>>(inputLen, inputDev, alpha, outputDev);
        cudaDeviceSynchronize();
    }

    void CudaKernels::LeakyReLUGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* outputDev, const float* outputGradientDev, float alpha, float* inputGradientDev)
    {
        leakyReluGrad<<<blocks, threads>>>(inputLen, outputDev, outputGradientDev, alpha, inputGradientDev);
        cudaDeviceSynchronize();
    }

    void CudaKernels::Div(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float v, float* outputDev)
    {
        div<<<blocks, threads>>>(inputLen, inputDev, v, outputDev);
        cudaDeviceSynchronize();
    }

    void CudaKernels::AdamStep(const dim3& blocks, const dim3& threads, int inputLen, float* parameterDev, float* gradientDev, float* mGradDev, float* vGradDev, float batchSize, float lr, float beta1, float beta2, float epsilon)
    {
        adamStep<<<blocks, threads>>>(inputLen, parameterDev, gradientDev, mGradDev, vGradDev, batchSize, lr, beta1, beta2, epsilon);
        cudaDeviceSynchronize();
    }

    void CudaKernels::SgdStep(const dim3& blocks, const dim3& threads, int inputLen, float* parameterDev, float* gradientDev, float batchSize, float lr)
    {
        sgdStep<<<blocks, threads>>>(inputLen, parameterDev, gradientDev, batchSize, lr);
        cudaDeviceSynchronize();
    }
}
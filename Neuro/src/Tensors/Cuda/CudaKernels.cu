#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Tensors/Cuda/CudaKernels.h"

__device__ int getIndex(int w, int h, int d, int n, int dim0, int dim0dim1, int dim0dim1dim2)
{
    return w + h * dim0 + d * dim0dim1 + n * dim0dim1dim2;
}

__device__ void getDims(int width, int height, int depth, int& dim0, int& dim0dim1, int& dim0dim1dim2)
{
    dim0 = width;
    dim0dim1 = width * height;
    dim0dim1dim2 = width * height * depth;
}

__global__ void upSample2D(const float* __restrict input, int inputWidth, int inputHeight, int inputDepth, int inputBatch, int scale, float* __restrict output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int inputLen = inputWidth * inputHeight * inputDepth * inputBatch;

    if (i >= inputLen)
        return;

    int inputDim0, inputDim0Dim1, inputDim0Dim1Dim2;
    getDims(inputWidth, inputHeight, inputDepth, inputDim0, inputDim0Dim1, inputDim0Dim1Dim2);

    int w = i % inputWidth;
    int h = (i / inputDim0) % inputHeight;
    int d = (i / inputDim0Dim1) % inputDepth;
    int n = i / inputDim0Dim1Dim2;

    int outputDim0 = inputDim0 * scale;
    int outputDim0Dim1 = inputDim0Dim1 * scale * scale;
    int outputDim0Dim1Dim2 = inputDim0Dim1Dim2 * scale * scale;

    for (int outH = h * scale; outH < (h + 1) * scale; ++outH)
    for (int outW = w * scale; outW < (w + 1) * scale; ++outW)
        output[getIndex(outW, outH, d, n, outputDim0, outputDim0Dim1, outputDim0Dim1Dim2)] = input[i];
}

__global__ void upSample2DGrad(const float* __restrict outputGrad, int scale, float* __restrict inputGrad, int inputWidth, int inputHeight, int inputDepth, int inputBatch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int inputLen = inputWidth * inputHeight * inputDepth * inputBatch;

    if (i >= inputLen)
        return;

    int inputDim0, inputDim0Dim1, inputDim0Dim1Dim2;
    getDims(inputWidth, inputHeight, inputDepth, inputDim0, inputDim0Dim1, inputDim0Dim1Dim2);

    int w = i % inputWidth;
    int h = (i / inputDim0) % inputHeight;
    int d = (i / inputDim0Dim1) % inputDepth;
    int n = i / inputDim0Dim1Dim2;

    int outputDim0 = inputDim0 * scale;
    int outputDim0Dim1 = inputDim0Dim1 * scale * scale;
    int outputDim0Dim1Dim2 = inputDim0Dim1Dim2 * scale * scale;

    for (int outH = h * scale; outH < (h + 1) * scale; ++outH)
    for (int outW = w * scale; outW < (w + 1) * scale; ++outW)
        inputGrad[i] += outputGrad[getIndex(outW, outH, d, n, outputDim0, outputDim0Dim1, outputDim0Dim1Dim2)];
}

__global__ void leakyRelu(int inputLen, const float* __restrict input, float alpha, float* __restrict output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputLen)
        output[i] = input[i] > 0 ? input[i] : (alpha * input[i]);
}

__global__ void leakyReluGrad(int inputLen, const float* __restrict output, const float* __restrict outputGrad, float alpha, float* __restrict inputGrad)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputLen)
        inputGrad[i] = (output[i] > 0 ? 1 : alpha) * outputGrad[i];
}

__global__ void div(int inputLen, const float* __restrict input, float v, float* __restrict output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputLen)
        output[i] = input[i] / v;
}

__global__ void addBroadcast(float alpha, const float* __restrict t1, int t1Width, int t1Height, int t1Depth, int t1Batch, float beta, const float* __restrict t2, int t2Width, int t2Height, int t2Depth, int t2Batch, float* __restrict output, int outputWidth, int outputHeight, int outputDepth, int outputBatch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int outputLen = outputWidth * outputHeight * outputDepth * outputBatch;

    if (i >= outputLen)
        return;
    
    int outputDim0, outputDim0Dim1, outputDim0Dim1Dim2;
    getDims(outputWidth, outputHeight, outputDepth, outputDim0, outputDim0Dim1, outputDim0Dim1Dim2);

    int t1Dim0, t1Dim0Dim1, t1Dim0Dim1Dim2;
    getDims(t1Width, t1Height, t1Depth, t1Dim0, t1Dim0Dim1, t1Dim0Dim1Dim2);

    int t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2;
    getDims(t2Width, t2Height, t2Depth, t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2);

    int w = i % outputWidth;
    int h = (i / outputDim0) % outputHeight;
    int d = (i / outputDim0Dim1) % outputDepth;
    int n = i / outputDim0Dim1Dim2;

    int t1N = n % t1Batch;
    int t2N = n % t2Batch;
    int t1D = d % t1Depth;
    int t2D = d % t2Depth;
    int t1H = h % t1Height;
    int t2H = h % t2Height;
    int t1W = w % t1Width;
    int t2W = w % t2Width;

    output[i] = alpha * t1[getIndex(t1W, t1H, t1D, t1N, t1Dim0, t1Dim0Dim1, t1Dim0Dim1Dim2)] + beta * t2[getIndex(t2W, t2H, t2D, t2N, t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2)];
}

template<int W, int H, int D, int N>
__global__ void sumTemplate(const float* __restrict input, int width, int height, int depth, int batch, float* __restrict output)
{
    //const size_t THREADS_PER_BLOCK = 1024;

    if (W && H && D && N)
    {
        // not implemented yet
    }
    else if (W && !H && !D && !N)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < height * depth * batch)
        {
            size_t tidx = idx * width;
            float tsum = 0;
            for (size_t i = 0; i < width; ++i)
            {
                tsum += input[tidx + i];
            }
            output[idx] = tsum;
        }
    }
    else if (!W && H && !D && !N)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < width * depth * batch)
        {
            size_t tidx = idx % width + (idx / width) * width * height;
            float tsum = 0;
            for (size_t i = 0; i < height; ++i)
            {
                tsum += input[tidx];
                tidx += width;
            }
            output[idx] = tsum;
        }
    }
    else if (!W && !H && D && !N)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < width * height * batch)
        {
            size_t tidx = idx % (width * height) + (idx / (width * height)) * width * height * depth;
            float tsum = 0;
            for (size_t i = 0; i < depth; i++)
            {
                tsum += input[tidx];
                tidx += width * height;
            }
            output[idx] = tsum;
        }
    }
    else if (!W && !H && !D && N)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < width * height * depth)
        {
            size_t tidx = idx;
            float tsum = 0;
            for (size_t i = 0; i < batch; i++)
            {
                tsum += input[tidx];
                tidx += width * height * depth;
            }
            output[idx] = tsum;
        }
    }
    else if (W && H && D && !N)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < batch)
        {
            size_t tidx = idx * (width * height * depth);
            float tsum = 0;
            for (size_t i = 0; i < width * height * depth; i++)
            {
                tsum += input[tidx + i];
            }
            output[idx] = tsum;
        }
    }
    else if (W && H && !D && N)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < depth)
        {
            size_t tidx = idx * (width * height);
            float tsum = 0;
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t i = 0; i < width * height; i++)
                {
                    tsum += input[tidx + i];
                }

                tidx += width * height * depth;
            }
            output[idx] = tsum;
        }
    }
    else if (!W && H && D && N)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < width)
        {
            size_t tidx = idx;
            float tsum = 0;
            for (size_t i = 0; i < height * depth * batch; i++)
            {
                tsum += input[tidx + i * width];
            }
            output[idx] = tsum;
        }
    }
}

template __global__ void sumTemplate<1, 1, 1, 1>(const float* __restrict input, int width, int height, int depth, int batch, float* __restrict output);
template __global__ void sumTemplate<1, 0, 0, 0>(const float* __restrict input, int width, int height, int depth, int batch, float* __restrict output);
template __global__ void sumTemplate<0, 1, 0, 0>(const float* __restrict input, int width, int height, int depth, int batch, float* __restrict output);
template __global__ void sumTemplate<0, 0, 1, 0>(const float* __restrict input, int width, int height, int depth, int batch, float* __restrict output);
template __global__ void sumTemplate<0, 0, 0, 1>(const float* __restrict input, int width, int height, int depth, int batch, float* __restrict output);
template __global__ void sumTemplate<1, 1, 1, 0>(const float* __restrict input, int width, int height, int depth, int batch, float* __restrict output);
template __global__ void sumTemplate<1, 1, 0, 1>(const float* __restrict input, int width, int height, int depth, int batch, float* __restrict output);
template __global__ void sumTemplate<0, 1, 1, 1>(const float* __restrict input, int width, int height, int depth, int batch, float* __restrict output);

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
__global__ void map(int inputLen, const float* __restrict input, F f, float* __restrict output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputLen)
        output[i] = f(input[i]);
}

namespace Neuro
{
    void CudaKernels::UpSample2D(const dim3& blocks, const dim3& threads, const float* inputDev, int inputWidth, int inputHeight, int inputDepth, int inputBatch, int scale, float* outputDev)
    {
        upSample2D<<<blocks, threads>>>(inputDev, inputWidth, inputHeight, inputDepth, inputBatch, scale, outputDev);
        cudaDeviceSynchronize();
    }

    void CudaKernels::UpSample2DGradient(const dim3& blocks, const dim3& threads, const float* outputGradientDev, int scale, float* inputGradientDev, int inputWidth, int inputHeight, int inputDepth, int inputBatch)
    {
        upSample2DGrad<<<blocks, threads>>>(outputGradientDev, scale, inputGradientDev, inputWidth, inputHeight, inputDepth, inputBatch);
        cudaDeviceSynchronize();
    }

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

    void CudaKernels::Sum(const dim3& blocks, const dim3& threads, const float* inputDev, int inputWidth, int inputHeight, int inputDepth, int inputBatch, int axis, float* outputDev)
    {
        if (axis == -1) // global
            sumTemplate<1, 1, 1, 1><<<blocks, threads>>>(inputDev, inputWidth, inputHeight, inputDepth, inputBatch, outputDev);
        else if (axis == 0) // width
            sumTemplate<1, 0, 0, 0><<<blocks, threads>>>(inputDev, inputWidth, inputHeight, inputDepth, inputBatch, outputDev);
        else if (axis == 1) // height
            sumTemplate<0, 1, 0, 0><<<blocks, threads>>>(inputDev, inputWidth, inputHeight, inputDepth, inputBatch, outputDev);
        else if (axis == 2) // depth
            sumTemplate<0, 0, 1, 0><<<blocks, threads>>>(inputDev, inputWidth, inputHeight, inputDepth, inputBatch, outputDev);
        else if (axis == 3) // batch
            sumTemplate<0, 0, 0, 1><<<blocks, threads>>>(inputDev, inputWidth, inputHeight, inputDepth, inputBatch, outputDev);
        else if (axis == 4) // 012
            sumTemplate<1, 1, 1, 0><<<blocks, threads>>>(inputDev, inputWidth, inputHeight, inputDepth, inputBatch, outputDev);
        else if (axis == 5) // 013
            sumTemplate<1, 1, 0, 1><<<blocks, threads>>>(inputDev, inputWidth, inputHeight, inputDepth, inputBatch, outputDev);
        else if (axis == 6) // 123
            sumTemplate<0, 1, 1, 1><<<blocks, threads>>>(inputDev, inputWidth, inputHeight, inputDepth, inputBatch, outputDev);
        cudaDeviceSynchronize();
    }

    void CudaKernels::AddBroadcast(const dim3& blocks, const dim3& threads, float alpha, const float* t1Dev, int t1Width, int t1Height, int t1Depth, int t1Batch, float beta, const float* t2Dev, int t2Width, int t2Height, int t2Depth, int t2Batch, float* outputDev, int outputWidth, int outputHeight, int outputDepth, int outputBatch)
    {
        addBroadcast<<<blocks, threads>>>(alpha, t1Dev, t1Width, t1Height, t1Depth, t1Batch, beta, t2Dev, t2Width, t2Height, t2Depth, t2Batch, outputDev, outputWidth, outputHeight, outputDepth, outputBatch);
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
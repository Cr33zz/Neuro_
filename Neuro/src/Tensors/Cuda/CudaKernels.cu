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

__global__ void setValue(int inputLen, float* __restrict input, float v, int subLen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int maxN = i * subLen + subLen;
    if (maxN > inputLen)
        maxN = inputLen;
    for (int n = i * subLen; n < maxN; ++n)
        input[n] = v;
}

__global__ void mul(int inputLen, const float* __restrict input, float v, float* __restrict output, int subLen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int maxN = i * subLen + subLen;
    if (maxN > inputLen)
        maxN = inputLen;
    for (int n = i * subLen; n < maxN; ++n)
        output[n] = input[n] * v;
}

__global__ void div(int inputLen, const float* __restrict input, float v, float* __restrict output, int subLen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int maxN = i * subLen + subLen;
    if (maxN > inputLen)
        maxN = inputLen;
    for (int n = i * subLen; n < maxN; ++n)
        output[n] = input[n] / v;
}

__global__ void pow(int inputLen, const float* __restrict input, float power, float* __restrict output, int subLen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int maxN = i * subLen + subLen;
    if (maxN > inputLen)
        maxN = inputLen;
    for (int n = i * subLen; n < maxN; ++n)
        output[n] = ::pow(input[n], power);
}

__global__ void powGrad(int inputLen, const float* __restrict input, float power, const float* __restrict outputGrad, float* __restrict inputGrad, int subLen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int maxN = i * subLen + subLen;
    if (maxN > inputLen)
        maxN = inputLen;
    if (power == 2)
    {
        for (int n = i * subLen; n < maxN; ++n)
            inputGrad[n] = outputGrad[n] * 2.f * input[n];
    }
    else
    {
        for (int n = i * subLen; n < maxN; ++n)
            inputGrad[n] = outputGrad[n] * power * ::pow(input[n], power - 1);
    }
}

__global__ void negate(int inputLen, const float* __restrict input, float* __restrict output, int subLen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int maxN = i * subLen + subLen;
    if (maxN > inputLen)
        maxN = inputLen;
    for (int n = i * subLen; n < maxN; ++n)
        output[n] = -input[n];
}

__global__ void inverse(int inputLen, const float* __restrict input, float* __restrict output, int subLen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int maxN = i * subLen + subLen;
    if (maxN > inputLen)
        maxN = inputLen;
    for (int n = i * subLen; n < maxN; ++n)
        output[n] = 1.f / input[n];
}

__global__ void add(int inputLen, const float* __restrict input, float v, float* __restrict output, int subLen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int maxN = i * subLen + subLen;
    if (maxN > inputLen)
        maxN = inputLen;
    for (int n = i * subLen; n < maxN; ++n)
        output[n] = input[n] + v;
}

__global__ void addBroadcast(float alpha, const float* __restrict t1, float beta, const float* __restrict t2, int t2Width, int t2Height, int t2Depth, int t2Batch, float* __restrict output, int outputWidth, int outputHeight, int outputDepth, int outputBatch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int outputLen = outputWidth * outputHeight * outputDepth * outputBatch;

    if (i >= outputLen)
        return;
    
    int outputDim0, outputDim0Dim1, outputDim0Dim1Dim2;
    getDims(outputWidth, outputHeight, outputDepth, outputDim0, outputDim0Dim1, outputDim0Dim1Dim2);

    int t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2;
    getDims(t2Width, t2Height, t2Depth, t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2);

    int w = t2Width == 1 ? 0 : (i % outputWidth);
    int h = t2Height == 1 ? 0 : ((i / outputDim0) % outputHeight);
    int d = t2Depth == 1 ? 0 : ((i / outputDim0Dim1) % outputDepth);
    int n = t2Batch == 1 ? 0 : (i / outputDim0Dim1Dim2);

    int t2N = t2Batch == 1 ? 0 : (n % t2Batch);
    int t2D = t2Depth == 1 ? 0 : (d % t2Depth);
    int t2H = t2Height == 1 ? 0 : (h % t2Height);
    int t2W = t2Width == 1 ? 0 : (w % t2Width);

    output[i] = alpha * t1[i] + beta * t2[getIndex(t2W, t2H, t2D, t2N, t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2)];
}

__global__ void mul(int len, const float* __restrict t1, const float* __restrict t2, float* __restrict output, int subLen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int maxN = i * subLen + subLen;
    if (maxN > len)
        maxN = len;
    for (int n = i * subLen; n < maxN; ++n)
        output[n] = t1[n] * t2[n];
}

__global__ void mulBroadcast(float alpha, const float* __restrict t1, float beta, const float* __restrict t2, int t2Width, int t2Height, int t2Depth, int t2Batch, float* __restrict output, int outputWidth, int outputHeight, int outputDepth, int outputBatch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int outputLen = outputWidth * outputHeight * outputDepth * outputBatch;

    if (i >= outputLen)
        return;

    int outputDim0, outputDim0Dim1, outputDim0Dim1Dim2;
    getDims(outputWidth, outputHeight, outputDepth, outputDim0, outputDim0Dim1, outputDim0Dim1Dim2);

    int t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2;
    getDims(t2Width, t2Height, t2Depth, t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2);

    int w = t2Width == 1 ? 0 : (i % outputWidth);
    int h = t2Height == 1 ? 0 : ((i / outputDim0) % outputHeight);
    int d = t2Depth == 1 ? 0 : ((i / outputDim0Dim1) % outputDepth);
    int n = t2Batch == 1 ? 0 : (i / outputDim0Dim1Dim2);

    int t2N = t2Batch == 1 ? 0 : (n % t2Batch);
    int t2D = t2Depth == 1 ? 0 : (d % t2Depth);
    int t2H = t2Height == 1 ? 0 : (h % t2Height);
    int t2W = t2Width == 1 ? 0 : (w % t2Width);

    output[i] = alpha * t1[i] * beta * t2[getIndex(t2W, t2H, t2D, t2N, t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2)];
}

__global__ void div(int len, float alpha, const float* __restrict t1, float beta, const float* __restrict t2, float* __restrict output, int subLen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int maxN = i * subLen + subLen;
    if (maxN > len)
        maxN = len;
    for (int n = i * subLen; n < maxN; ++n)
        output[n] = (alpha * t1[n]) / (beta * t2[n]);
}

__global__ void divBroadcast(float alpha, const float* __restrict t1, int t1Width, int t1Height, int t1Depth, int t1Batch, float beta, const float* __restrict t2, int t2Width, int t2Height, int t2Depth, int t2Batch, float* __restrict output, int outputWidth, int outputHeight, int outputDepth, int outputBatch)
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

    output[i] = (alpha * t1[getIndex(t1W, t1H, t1D, t1N, t1Dim0, t1Dim0Dim1, t1Dim0Dim1Dim2)]) / (beta * t2[getIndex(t2W, t2H, t2D, t2N, t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2)]);
}

template<int W, int H, int D, int N>
__global__ void sumTemplate(const float* __restrict input, int width, int height, int depth, int batch, float* __restrict output)
{        
    const size_t THREADS_PER_BLOCK = 1024;

    if (W && H && D && N)
    {
        __shared__ float sdata[THREADS_PER_BLOCK];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

        float mySum = (i < width) ? input[i] : 0;

        if (i + blockDim.x < width)
            mySum += input[i + blockDim.x];

        sdata[tid] = mySum;
        __syncthreads();

        // do reduction in shared mem
        for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
        {
            if (tid < s)
            {
                sdata[tid] = mySum = mySum + sdata[tid + s];
            }

            __syncthreads();
        }

        // write result for this block to global mem
        if (tid == 0) output[blockIdx.x] = mySum;
    }
    else if (W && !H && !D && !N)
    {
        __shared__ float sdata[THREADS_PER_BLOCK];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.y * (blockDim.x * 2) + threadIdx.x;
        unsigned int offset = blockIdx.x * width;

        float mySum = (i < width) ? input[offset + i] : 0;

        if (i + blockDim.x < width)
            mySum += input[offset + i + blockDim.x];

        sdata[tid] = mySum;
        __syncthreads();

        // do reduction in shared mem
        for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
        {
            if (tid < s)
            {
                sdata[tid] = mySum = mySum + sdata[tid + s];
            }

            __syncthreads();
        }

        // write result for this block to global mem
        if (tid == 0) output[blockIdx.x * gridDim.y + blockIdx.y] = mySum;
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
    else if (W && H && !D && !N)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < depth * batch)
        {
            size_t tidx = idx * (width * height);
            float tsum = 0;
            for (size_t i = 0; i < width * height; i++)
            {
                tsum += input[tidx + i];
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
template __global__ void sumTemplate<1, 1, 0, 0>(const float* __restrict input, int width, int height, int depth, int batch, float* __restrict output);
template __global__ void sumTemplate<1, 1, 1, 0>(const float* __restrict input, int width, int height, int depth, int batch, float* __restrict output);
template __global__ void sumTemplate<1, 1, 0, 1>(const float* __restrict input, int width, int height, int depth, int batch, float* __restrict output);
template __global__ void sumTemplate<0, 1, 1, 1>(const float* __restrict input, int width, int height, int depth, int batch, float* __restrict output);

__global__ void adamStep(int inputLen, float* __restrict parameterDev, const float* __restrict gradientDev, float* __restrict mGradDev, float* __restrict vGradDev, /*float batchSize, */float lr, float beta1, float beta2, float epsilon)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputLen)
    {
        float grad = gradientDev[i]/* / batchSize*/;
        mGradDev[i] = beta1 * mGradDev[i] + (1 - beta1) * grad;
        vGradDev[i] = beta2 * vGradDev[i] + (1 - beta2) * grad * grad;
        parameterDev[i] -= mGradDev[i] / (sqrt(vGradDev[i]) + epsilon) * lr;
    }
}

__global__ void sgdStep(int inputLen, float* __restrict parameterDev, const float* __restrict gradientDev, /*float batchSize, */float lr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputLen)
    {
        parameterDev[i] -= gradientDev[i]/* / batchSize*/ * lr;
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
    void CudaKernels::One(const dim3& blocks, const dim3& threads, int inputLen, float* inputDev, int subLen)
    {
        setValue<<<blocks, threads>>>(inputLen, inputDev, 1, subLen);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::UpSample2D(const dim3& blocks, const dim3& threads, const float* inputDev, int inputWidth, int inputHeight, int inputDepth, int inputBatch, int scale, float* outputDev)
    {
        upSample2D<<<blocks, threads>>>(inputDev, inputWidth, inputHeight, inputDepth, inputBatch, scale, outputDev);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::UpSample2DGradient(const dim3& blocks, const dim3& threads, const float* outputGradientDev, int scale, float* inputGradientDev, int inputWidth, int inputHeight, int inputDepth, int inputBatch)
    {
        upSample2DGrad<<<blocks, threads>>>(outputGradientDev, scale, inputGradientDev, inputWidth, inputHeight, inputDepth, inputBatch);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::LeakyReLU(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float alpha, float* outputDev)
    {
        leakyRelu<<<blocks, threads>>>(inputLen, inputDev, alpha, outputDev);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::LeakyReLUGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* outputDev, const float* outputGradientDev, float alpha, float* inputGradientDev)
    {
        leakyReluGrad<<<blocks, threads>>>(inputLen, outputDev, outputGradientDev, alpha, inputGradientDev);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::Mul(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float v, float* outputDev, int subLen)
    {
        mul<<<blocks, threads>>>(inputLen, inputDev, v, outputDev, subLen);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::Div(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float v, float* outputDev, int subLen)
    {
        div<<<blocks, threads>>>(inputLen, inputDev, v, outputDev, subLen);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::Add(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float v, float* outputDev, int subLen)
    {
        add<<<blocks, threads>>>(inputLen, inputDev, v, outputDev, subLen);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::Pow(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float power, float* outputDev, int subLen)
    {
        pow<<<blocks, threads>>>(inputLen, inputDev, power, outputDev, subLen);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::PowGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float power, const float* outputGradientDev, float* inputGradientDev, int subLen)
    {
        powGrad<<<blocks, threads>>>(inputLen, inputDev, power, outputGradientDev, inputGradientDev, subLen);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::Negate(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float* outputDev, int subLen)
    {
        negate<<<blocks, threads>>>(inputLen, inputDev, outputDev, subLen);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::Inverse(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float* outputDev, int subLen)
    {
        inverse<<<blocks, threads>>>(inputLen, inputDev, outputDev, subLen);
        cudaStreamSynchronize(0);
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
        else if (axis == 4) // 01
            sumTemplate<1, 1, 0, 0><<<blocks, threads>>>(inputDev, inputWidth, inputHeight, inputDepth, inputBatch, outputDev);
        else if (axis == 5) // 012
            sumTemplate<1, 1, 1, 0><<<blocks, threads>>>(inputDev, inputWidth, inputHeight, inputDepth, inputBatch, outputDev);
        else if (axis == 6) // 013
            sumTemplate<1, 1, 0, 1><<<blocks, threads>>>(inputDev, inputWidth, inputHeight, inputDepth, inputBatch, outputDev);
        else if (axis == 7) // 123
            sumTemplate<0, 1, 1, 1><<<blocks, threads>>>(inputDev, inputWidth, inputHeight, inputDepth, inputBatch, outputDev);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::AddBroadcast(const dim3& blocks, const dim3& threads, float alpha, const float* t1Dev, float beta, const float* t2Dev, int t2Width, int t2Height, int t2Depth, int t2Batch, float* outputDev, int outputWidth, int outputHeight, int outputDepth, int outputBatch)
    {
        addBroadcast<<<blocks, threads>>>(alpha, t1Dev, beta, t2Dev, t2Width, t2Height, t2Depth, t2Batch, outputDev, outputWidth, outputHeight, outputDepth, outputBatch);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::Mul(const dim3& blocks, const dim3& threads, int len, const float* t1, const float* t2, float* outputDev, int subLen)
    {
        mul<<<blocks, threads>>>(len, t1, t2, outputDev, subLen);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::MulBroadcast(const dim3& blocks, const dim3& threads, float alpha, const float* t1Dev, float beta, const float* t2Dev, int t2Width, int t2Height, int t2Depth, int t2Batch, float* outputDev, int outputWidth, int outputHeight, int outputDepth, int outputBatch)
    {
        mulBroadcast<<<blocks, threads>>>(alpha, t1Dev, beta, t2Dev, t2Width, t2Height, t2Depth, t2Batch, outputDev, outputWidth, outputHeight, outputDepth, outputBatch);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::Div(const dim3& blocks, const dim3& threads, int len, float alpha, const float* t1, float beta, const float* t2, float* outputDev, int subLen)
    {
        div<<<blocks, threads>>>(len, alpha, t1, beta,t2, outputDev, subLen);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::DivBroadcast(const dim3& blocks, const dim3& threads, float alpha, const float* t1Dev, int t1Width, int t1Height, int t1Depth, int t1Batch, float beta, const float* t2Dev, int t2Width, int t2Height, int t2Depth, int t2Batch, float* outputDev, int outputWidth, int outputHeight, int outputDepth, int outputBatch)
    {
        divBroadcast<<<blocks, threads>>>(alpha, t1Dev, t1Width, t1Height, t1Depth, t1Batch, beta, t2Dev, t2Width, t2Height, t2Depth, t2Batch, outputDev, outputWidth, outputHeight, outputDepth, outputBatch);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::AdamStep(const dim3& blocks, const dim3& threads, int inputLen, float* parameterDev, const float* gradientDev, float* mGradDev, float* vGradDev, /*float batchSize, */float lr, float beta1, float beta2, float epsilon)
    {
        adamStep<<<blocks, threads>>>(inputLen, parameterDev, gradientDev, mGradDev, vGradDev, /*batchSize, */lr, beta1, beta2, epsilon);
        cudaStreamSynchronize(0);
    }

    void CudaKernels::SgdStep(const dim3& blocks, const dim3& threads, int inputLen, float* parameterDev, const float* gradientDev, /*float batchSize, */float lr)
    {
        sgdStep<<<blocks, threads>>>(inputLen, parameterDev, gradientDev, /*batchSize, */lr);
        cudaStreamSynchronize(0);
    }
}
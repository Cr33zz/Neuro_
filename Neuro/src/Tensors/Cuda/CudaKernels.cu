#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "Tensors/Cuda/CudaKernels.h"

__device__ int sign(float x)
{
    int t = x < 0 ? -1 : 0;
    return x > 0 ? 1 : t;
}

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

__global__ void dropout(int inputLen, const float* __restrict input, float prob, float* __restrict mask, float* __restrict output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputLen; i += gridDim.x * blockDim.x)
    {
        mask[i] = (mask[i] < prob ? 0.f : 1.f) / prob;
        output[i] = input[i] * mask[i];
    }
}

__global__ void dropoutGrad(int inputLen, const float* __restrict outputGrad, const float* __restrict mask, float* __restrict inputGrad)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputLen; i += gridDim.x * blockDim.x)
        inputGrad[i] = outputGrad[i] * mask[i];
}

__global__ void transpose(int inputLen, const float* __restrict input, int axis0, int axis1, int axis2, int axis3, int stride0, int stride1, int stride2, int stride3, float* __restrict output, int outputStride1, int outputStride2, int outputStride3)
{
    int stride[4] = { stride0, stride1, stride2, stride3 };
    int outputHeight = outputStride2 / outputStride1;
    int outputDepth = outputStride3 / outputStride2;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputLen; i += gridDim.x * blockDim.x)
    {
        int w = i % outputStride1;
        int h = (i / outputStride1) % outputHeight;
        int d = (i / outputStride2) % outputDepth;
        int n = i / outputStride3;

        int index = w * stride[axis0] + h * stride[axis1] + d * stride[axis2] + n * stride[axis3];
        output[i] = input[index];
    }
}

__global__ void extractSubTensor2D(int outputLen, const float* __restrict input, int inputStride1, int inputStride2, int inputStride3, int widthOffset, int heightOffset, float* __restrict output, int outputStride1, int outputStride2, int outputStride3)
{
    int outputHeight = outputStride2 / outputStride1;
    int outputDepth = outputStride3 / outputStride2;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < outputLen; i += gridDim.x * blockDim.x)
    {
        int w = i % outputStride1;
        int h = (i / outputStride1) % outputHeight;
        int d = (i / outputStride2) % outputDepth;
        int n = i / outputStride3;

        output[i] = input[getIndex(w + widthOffset, h + heightOffset, d, n, inputStride1, inputStride2, inputStride3)];
    }
}

__global__ void fuseSubTensor2D(int inputLen, const float* __restrict input, int inputStride1, int inputStride2, int inputStride3, int widthOffset, int heightOffset, float* __restrict output, int outputStride1, int outputStride2, int outputStride3)
{
    int inputHeight = inputStride2 / inputStride1;
    int inputDepth = inputStride3 / inputStride2;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputLen; i += gridDim.x * blockDim.x)
    {
        int w = i % inputStride1;
        int h = (i / inputStride1) % inputHeight;
        int d = (i / inputStride2) % inputDepth;
        int n = i / inputStride3;

        output[getIndex(w + widthOffset, h + heightOffset, d, n, outputStride1, outputStride2, outputStride3)] = input[i];
    }
}

__global__ void constantPad2D(int outputLen, const float* __restrict input, int inputStride1, int inputStride2, int inputStride3, int left, int right, int top, int bottom, float value, float* __restrict output, int outputStride1, int outputStride2, int outputStride3)
{
    int inputHeight = inputStride2 / inputStride1;
    int outputHeight = outputStride2 / outputStride1;
    int outputDepth = outputStride3 / outputStride2;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < outputLen; i += gridDim.x * blockDim.x)
    {
        float v = value;
        int w = i % outputStride1;
        int h = (i / outputStride1) % outputHeight;
        int d = (i / outputStride2) % outputDepth;
        int n = i / outputStride3;

        if (w >= left && h >= top && w < inputStride1 + left && h < inputHeight + top)
            v = input[getIndex(w - left, h - top, d, n, inputStride1, inputStride2, inputStride3)];

        output[i] = v;
    }
}

__global__ void reflectPad2D(int outputLen, const float* __restrict input, int inputStride1, int inputStride2, int inputStride3, int left, int right, int top, int bottom, float* __restrict output, int outputStride1, int outputStride2, int outputStride3)
{
    int inputWidth = inputStride1;
    int inputHeight = inputStride2 / inputStride1;
    int outputHeight = outputStride2 / outputStride1;
    int outputDepth = outputStride3 / outputStride2;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < outputLen; i += gridDim.x * blockDim.x)
    {
        int w = i % outputStride1;
        int h = (i / outputStride1) % outputHeight;
        int d = (i / outputStride2) % outputDepth;
        int n = i / outputStride3;

        int inputW = w - left;
        int inputH = h - top;

        if (inputW < 0)
            inputW = -inputW;
        else if (inputW >= inputWidth)
            inputW = abs(inputWidth - inputW);
        inputW %= inputWidth;

        if (inputH < 0)
            inputH = -inputH;
        else if (inputH >= inputHeight)
            inputH = abs(inputHeight - inputH);
        inputH %= inputHeight;

        output[i] = input[getIndex(inputW, inputH, d, n, inputStride1, inputStride2, inputStride3)];
    }
}

__global__ void pad2DGrad(int inputGradLen, const float* __restrict outputGrad, int outputGradStride1, int outputGradStride2, int outputGradStride3, int left, int right, int top, int bottom, float* __restrict inputGrad, int inputGradStride1, int inputGradStride2, int inputGradStride3)
{
    int inputGradHeight = inputGradStride2 / inputGradStride1;
    int inputGradDepth = inputGradStride3 / inputGradStride2;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputGradLen; i += gridDim.x * blockDim.x)
    {
        int w = i % inputGradStride1;
        int h = (i / inputGradStride1) % inputGradHeight;
        int d = (i / inputGradStride2) % inputGradDepth;
        int n = i / inputGradStride3;

        inputGrad[i] = outputGrad[getIndex(w + left, h + top, d, n, outputGradStride1, outputGradStride2, outputGradStride3)];
    }
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
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputLen; i += gridDim.x * blockDim.x)
        output[i] = input[i] > 0 ? input[i] : (alpha * input[i]);
}

__global__ void leakyReluGrad(int inputLen, const float* __restrict output, const float* __restrict outputGrad, float alpha, float* __restrict inputGrad)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputLen; i += gridDim.x * blockDim.x)
        inputGrad[i] = (output[i] > 0 ? 1 : alpha) * outputGrad[i];
}

__global__ void pow(int inputLen, const float* __restrict input, float power, float* __restrict output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputLen; i += gridDim.x * blockDim.x)
        output[i] = ::pow(input[i], power);
}

__global__ void powGrad(int inputLen, const float* __restrict input, float power, const float* __restrict outputGrad, float* __restrict inputGrad)
{
    if (power == 2)
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputLen; i += gridDim.x * blockDim.x)
            inputGrad[i] = outputGrad[i] * 2.f * input[i];
    }
    else
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputLen; i += gridDim.x * blockDim.x)
            inputGrad[i] = outputGrad[i] * power * ::pow(input[i], power - 1);
    }
}

__global__ void abs(int inputLen, const float* __restrict input, float* __restrict output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputLen; i += gridDim.x * blockDim.x)
        output[i] = ::abs(input[i]);
}

__global__ void absGrad(int inputLen, const float* __restrict input, const float* __restrict outputGrad, float* __restrict inputGrad)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputLen; i += gridDim.x * blockDim.x)
        inputGrad[i] = sign(input[i]) * outputGrad[i];
}

__global__ void clip(int inputLen, const float* __restrict input, float min, float max, float* __restrict output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputLen; i += gridDim.x * blockDim.x)
        output[i] = input[i] < min ? min : (input[i] > max ? max : input[i]);
}

__global__ void clipGrad(int inputLen, const float* __restrict input, float min, float max, const float* __restrict outputGrad, float* __restrict inputGrad)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputLen; i += gridDim.x * blockDim.x)
        inputGrad[i] = (input[i] >= min && input[i] <= max) ? outputGrad[i] : 0;
}

__global__ void negate(int inputLen, const float* __restrict input, float* __restrict output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputLen; i += gridDim.x * blockDim.x)
        output[i] = -input[i];
}

__global__ void inverse(int inputLen, const float* __restrict input, float alpha, float* __restrict output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputLen; i += gridDim.x * blockDim.x)
        output[i] = __fdividef(alpha, input[i]);
}

__global__ void log(int inputLen, const float* __restrict input, float* __restrict output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputLen; i += gridDim.x * blockDim.x)
        output[i] = ::log(input[i]);
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

__global__ void addBroadcast(float alpha, const float* __restrict t1, int t1Width, int t1Height, int t1Depth, int t1Batch, float beta, const float* __restrict t2, int t2Width, int t2Height, int t2Depth, int t2Batch, float* __restrict output, int outputWidth, int outputHeight, int outputDepth, int outputBatch)
{
    int outputLen = outputWidth * outputHeight * outputDepth * outputBatch;

    int outputDim0, outputDim0Dim1, outputDim0Dim1Dim2;
    getDims(outputWidth, outputHeight, outputDepth, outputDim0, outputDim0Dim1, outputDim0Dim1Dim2);

    int t1Dim0, t1Dim0Dim1, t1Dim0Dim1Dim2;
    getDims(t1Width, t1Height, t1Depth, t1Dim0, t1Dim0Dim1, t1Dim0Dim1Dim2);

    int t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2;
    getDims(t2Width, t2Height, t2Depth, t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < outputLen; i += gridDim.x * blockDim.x)
    {
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

__global__ void mulBroadcast(float alpha, const float* __restrict t1, int t1Width, int t1Height, int t1Depth, int t1Batch, float beta, const float* __restrict t2, int t2Width, int t2Height, int t2Depth, int t2Batch, float* __restrict output, int outputWidth, int outputHeight, int outputDepth, int outputBatch)
{
    int outputLen = outputWidth * outputHeight * outputDepth * outputBatch;

    int outputDim0, outputDim0Dim1, outputDim0Dim1Dim2;
    getDims(outputWidth, outputHeight, outputDepth, outputDim0, outputDim0Dim1, outputDim0Dim1Dim2);

    int t1Dim0, t1Dim0Dim1, t1Dim0Dim1Dim2;
    getDims(t1Width, t1Height, t1Depth, t1Dim0, t1Dim0Dim1, t1Dim0Dim1Dim2);

    int t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2;
    getDims(t2Width, t2Height, t2Depth, t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < outputLen; i += gridDim.x * blockDim.x)
    {
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

        output[i] = alpha * t1[getIndex(t1W, t1H, t1D, t1N, t1Dim0, t1Dim0Dim1, t1Dim0Dim1Dim2)] * beta * t2[getIndex(t2W, t2H, t2D, t2N, t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2)];
    }
}

__global__ void div(int len, float alpha, const float* __restrict t1, float beta, const float* __restrict t2, float* __restrict output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x)
        output[i] = __fdividef(alpha * t1[i], beta * t2[i]);
}

__global__ void divBroadcast(float alpha, const float* __restrict t1, int t1Width, int t1Height, int t1Depth, int t1Batch, float beta, const float* __restrict t2, int t2Width, int t2Height, int t2Depth, int t2Batch, float* __restrict output, int outputWidth, int outputHeight, int outputDepth, int outputBatch)
{
    int outputLen = outputWidth * outputHeight * outputDepth * outputBatch;

    int outputDim0, outputDim0Dim1, outputDim0Dim1Dim2;
    getDims(outputWidth, outputHeight, outputDepth, outputDim0, outputDim0Dim1, outputDim0Dim1Dim2);

    int t1Dim0, t1Dim0Dim1, t1Dim0Dim1Dim2;
    getDims(t1Width, t1Height, t1Depth, t1Dim0, t1Dim0Dim1, t1Dim0Dim1Dim2);

    int t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2;
    getDims(t2Width, t2Height, t2Depth, t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < outputLen; i += gridDim.x * blockDim.x)
    {
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

        output[i] = __fdividef(alpha * t1[getIndex(t1W, t1H, t1D, t1N, t1Dim0, t1Dim0Dim1, t1Dim0Dim1Dim2)], beta * t2[getIndex(t2W, t2H, t2D, t2N, t2Dim0, t2Dim0Dim1, t2Dim0Dim1Dim2)]);
    }
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
    void CudaKernels::ExtractSubTensor2D(const dim3& blocks, const dim3& threads, int outputLen, const float* inputDev, int inputStride1, int inputStride2, int inputStride3, int widthOffset, int heightOffset, float* __restrict outputDev, int outputStride1, int outputStride2, int outputStride3)
    {
        extractSubTensor2D<<<blocks, threads>>>(outputLen, inputDev, inputStride1, inputStride2, inputStride3, widthOffset, heightOffset, outputDev, outputStride1, outputStride2, outputStride3);
    }

    void CudaKernels::FuseSubTensor2D(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, int inputStride1, int inputStride2, int inputStride3, int widthOffset, int heightOffset, float* __restrict outputDev, int outputStride1, int outputStride2, int outputStride3)
    {
        fuseSubTensor2D<<<blocks, threads>>>(inputLen, inputDev, inputStride1, inputStride2, inputStride3, widthOffset, heightOffset, outputDev, outputStride1, outputStride2, outputStride3);
    }

    void CudaKernels::ConstantPad2D(const dim3& blocks, const dim3& threads, int outputLen, const float* inputDev, int inputStride1, int inputStride2, int inputStride3, int left, int right, int top, int bottom, float value, float* __restrict outputDev, int outputStride1, int outputStride2, int outputStride3)
    {
        constantPad2D<<<blocks, threads>>>(outputLen, inputDev, inputStride1, inputStride2, inputStride3, left, right, top, bottom, value, outputDev, outputStride1, outputStride2, outputStride3);
    }

    void CudaKernels::ReflectPad2D(const dim3& blocks, const dim3& threads, int outputLen, const float* inputDev, int inputStride1, int inputStride2, int inputStride3, int left, int right, int top, int bottom, float* __restrict outputDev, int outputStride1, int outputStride2, int outputStride3)
    {
        reflectPad2D<<<blocks, threads>>>(outputLen, inputDev, inputStride1, inputStride2, inputStride3, left, right, top, bottom, outputDev, outputStride1, outputStride2, outputStride3);
    }

    void CudaKernels::Pad2DGradient(const dim3& blocks, const dim3& threads, int inputGradLen, const float* outputGradDev, int outputGradStride1, int outputGradStride2, int outputGradStride3, int left, int right, int top, int bottom, float* inputGradDev, int inputGradStride1, int inputGradStride2, int inputGradStride3)
    {
        pad2DGrad << <blocks, threads >> > (inputGradLen, outputGradDev, outputGradStride1, outputGradStride2, outputGradStride3, left, right, top, bottom, inputGradDev, inputGradStride1, inputGradStride2, inputGradStride3);
    }

    void CudaKernels::UpSample2D(const dim3& blocks, const dim3& threads, const float* inputDev, int inputWidth, int inputHeight, int inputDepth, int inputBatch, int scale, float* outputDev)
    {
        upSample2D<<<blocks, threads>>>(inputDev, inputWidth, inputHeight, inputDepth, inputBatch, scale, outputDev);
    }

    void CudaKernels::UpSample2DGradient(const dim3& blocks, const dim3& threads, const float* outputGradientDev, int scale, float* inputGradientDev, int inputWidth, int inputHeight, int inputDepth, int inputBatch)
    {
        upSample2DGrad<<<blocks, threads>>>(outputGradientDev, scale, inputGradientDev, inputWidth, inputHeight, inputDepth, inputBatch);
    }

    void CudaKernels::LeakyReLU(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float alpha, float* outputDev)
    {
        leakyRelu<<<blocks, threads>>>(inputLen, inputDev, alpha, outputDev);
    }

    void CudaKernels::LeakyReLUGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* outputDev, const float* outputGradientDev, float alpha, float* inputGradientDev)
    {
        leakyReluGrad<<<blocks, threads>>>(inputLen, outputDev, outputGradientDev, alpha, inputGradientDev);
    }

    void CudaKernels::Add(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float v, float* outputDev, int subLen)
    {
        add<<<blocks, threads>>>(inputLen, inputDev, v, outputDev, subLen);
    }

    void CudaKernels::Pow(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float power, float* outputDev)
    {
        pow<<<blocks, threads>>>(inputLen, inputDev, power, outputDev);
    }

    void CudaKernels::PowGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float power, const float* outputGradientDev, float* inputGradientDev)
    {
        powGrad<<<blocks, threads>>>(inputLen, inputDev, power, outputGradientDev, inputGradientDev);
    }

    void CudaKernels::Clip(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float min, float max, float* outputDev)
    {
        clip<<<blocks, threads>>>(inputLen, inputDev, min, max, outputDev);
    }

    void CudaKernels::ClipGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float min, float max, const float* outputGradientDev, float* inputGradientDev)
    {
        clipGrad<<<blocks, threads>>>(inputLen, inputDev, min, max, outputGradientDev, inputGradientDev);
    }
    
    void CudaKernels::Abs(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float* outputDev)
    {
        abs<<<blocks, threads>>>(inputLen, inputDev, outputDev);
    }

    void CudaKernels::AbsGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, const float* outputGradientDev, float* inputGradientDev)
    {
        absGrad<<<blocks, threads>>>(inputLen, inputDev, outputGradientDev, inputGradientDev);
    }

    void CudaKernels::Negate(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float* outputDev)
    {
        negate<<<blocks, threads>>>(inputLen, inputDev, outputDev);
    }

    void CudaKernels::Log(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float* outputDev)
    {
        log<<<blocks, threads>>>(inputLen, inputDev, outputDev);
    }

    void CudaKernels::Inverse(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float alpha, float* outputDev)
    {
        inverse<<<blocks, threads>>>(inputLen, inputDev, alpha, outputDev);
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
    }

    void CudaKernels::AddBroadcast(const dim3& blocks, const dim3& threads, float alpha, const float* t1Dev, int t1Width, int t1Height, int t1Depth, int t1Batch, float beta, const float* t2Dev, int t2Width, int t2Height, int t2Depth, int t2Batch, float* outputDev, int outputWidth, int outputHeight, int outputDepth, int outputBatch)
    {
        addBroadcast<<<blocks, threads>>>(alpha, t1Dev, t1Width, t1Height, t1Depth, t1Batch, beta, t2Dev, t2Width, t2Height, t2Depth, t2Batch, outputDev, outputWidth, outputHeight, outputDepth, outputBatch);
    }

    void CudaKernels::Mul(const dim3& blocks, const dim3& threads, int len, const float* t1, const float* t2, float* outputDev, int subLen)
    {
        mul<<<blocks, threads>>>(len, t1, t2, outputDev, subLen);
    }

    void CudaKernels::MulBroadcast(const dim3& blocks, const dim3& threads, float alpha, const float* t1Dev, int t1Width, int t1Height, int t1Depth, int t1Batch, float beta, const float* t2Dev, int t2Width, int t2Height, int t2Depth, int t2Batch, float* outputDev, int outputWidth, int outputHeight, int outputDepth, int outputBatch)
    {
        mulBroadcast<<<blocks, threads>>>(alpha, t1Dev, t1Width, t1Height, t1Depth, t1Batch, beta, t2Dev, t2Width, t2Height, t2Depth, t2Batch, outputDev, outputWidth, outputHeight, outputDepth, outputBatch);
    }

    void CudaKernels::Div(const dim3& blocks, const dim3& threads, int len, float alpha, const float* t1, float beta, const float* t2, float* outputDev)
    {
        div<<<blocks, threads>>>(len, alpha, t1, beta,t2, outputDev);
    }

    void CudaKernels::DivBroadcast(const dim3& blocks, const dim3& threads, float alpha, const float* t1Dev, int t1Width, int t1Height, int t1Depth, int t1Batch, float beta, const float* t2Dev, int t2Width, int t2Height, int t2Depth, int t2Batch, float* outputDev, int outputWidth, int outputHeight, int outputDepth, int outputBatch)
    {
        divBroadcast<<<blocks, threads>>>(alpha, t1Dev, t1Width, t1Height, t1Depth, t1Batch, beta, t2Dev, t2Width, t2Height, t2Depth, t2Batch, outputDev, outputWidth, outputHeight, outputDepth, outputBatch);
    }

    void CudaKernels::AdamStep(const dim3& blocks, const dim3& threads, int inputLen, float* parameterDev, const float* gradientDev, float* mGradDev, float* vGradDev, float lr, float beta1, float beta2, float epsilon)
    {
        adamStep<<<blocks, threads>>>(inputLen, parameterDev, gradientDev, mGradDev, vGradDev, lr, beta1, beta2, epsilon);
    }

    void CudaKernels::SgdStep(const dim3& blocks, const dim3& threads, int inputLen, float* parameterDev, const float* gradientDev, float lr)
    {
        sgdStep<<<blocks, threads>>>(inputLen, parameterDev, gradientDev, lr);
    }

    void CudaKernels::Dropout(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float prob, float* maskDev, float* outputDev)
    {
        dropout<<<blocks, threads>>>(inputLen, inputDev, prob, maskDev, outputDev);
    }

    void CudaKernels::DropoutGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* outputGradDev, const float* maskDev, float* inputGradDev)
    {
        dropoutGrad<<<blocks, threads>>>(inputLen, outputGradDev, maskDev, inputGradDev);
    }

    void CudaKernels::Transpose(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, int axis0, int axis1, int axis2, int axis3, int stride0, int stride1, int stride2, int stride3, float* outputDev, int outputStride1, int outputStride2, int outputStride3)
    {
        transpose<<<blocks, threads>>>(inputLen, inputDev, axis0, axis1, axis2, axis3, stride0, stride1, stride2, stride3, outputDev, outputStride1, outputStride2, outputStride3);
    }
}
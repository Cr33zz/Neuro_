#pragma once
#include "Types.h"

#ifdef CUDA_ENABLED
#include <cuda.h>
#include <cudnn.h>
#include <cublas.h>
#endif

#include "Tensors/TensorOpMultiCpu.h"

namespace Neuro
{
    class TensorOpGpu : public TensorOpMultiCpu
    {
#ifdef CUDA_ENABLED
    public:
        TensorOpGpu();

        virtual void Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& result) const override;
        virtual void Mul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& result) const override;
        virtual void Transpose(const Tensor& t, Tensor& result) const override;
        virtual void Conv2D(const Tensor& t, const Tensor& kernels, int stride, EPaddingMode padding, Tensor& result) const override;
        virtual void Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, int stride, EPaddingMode padding, Tensor& inputGradients) const override;
        virtual void Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, int stride, EPaddingMode padding, Tensor& kernelsGradient) const override;
        virtual void Pool(const Tensor& t, int filterSize, int stride, EPoolingMode type, int paddingX, int paddingY, Tensor& result) const override;
        virtual void PoolGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, int filterSize, int stride, EPoolingMode type, int paddingX, int paddingY, Tensor& result) const override;
        virtual void SumBatches(const Tensor& t, Tensor& result) const override;
        virtual void Elu(const Tensor& input, float alpha, Tensor& result) const override;
        virtual void EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& result) const override;
        virtual void Softmax(const Tensor& input, Tensor& result) const override;
        virtual void SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& result) const override;

    private:
        static cudnnPoolingMode_t GetCudnnPoolType(EPoolingMode type);
        static void GetKernelRunParams(int count, dim3& blocks, dim3& threads);
        static int GetBlocksNum(int count);

        static bool s_Initialized;
        static cudaDeviceProp s_CudaDevProp;
        static cublasHandle_t s_CublasHandle;
        static cudnnHandle_t s_CudnnHandle;
#endif
    };
}
#pragma once
#include <cuda.h>
#include <cudnn.h>
#include <cublas.h>

#include "Types.h"
#include "Tensors/TensorOpMultiCpu.h"

namespace Neuro
{
    class TensorOpGpu : public TensorOpMultiCpu
    {
    public:
        TensorOpGpu();

        virtual void Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& output) const override;
        virtual void Mul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const override;
        virtual void Div(const Tensor& input, float v, Tensor& output) const override;
        virtual void Sum(const Tensor& input, EAxis axis, int batch, Tensor& output) const override;
        virtual void Transpose(const Tensor& input, Tensor& output) const override;
        virtual void Conv2D(const Tensor& input, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& output) const override;
        virtual void Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& inputGradient) const override;
        virtual void Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& kernelsGradient) const override;
        virtual void Pool2D(const Tensor& t, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, Tensor& output) const override;
        virtual void Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, Tensor& inputGradient) const override;
        virtual void BatchNormalization(const Tensor& t, const Tensor& gamma, const Tensor& beta, const Tensor& runningMean, const Tensor& runningVar, Tensor& output) const override;
        virtual void BatchNormalizationTrain(const Tensor& t, const Tensor& gamma, const Tensor& beta, float momentum, Tensor& runningMean, Tensor& runningVar, Tensor& saveMean, Tensor& saveInvVariance, Tensor& output) const override;
        virtual void BatchNormalizationGradient(const Tensor& input, const Tensor& gamma, const Tensor& outputGradient, const Tensor& savedMean, const Tensor& savedInvVariance, Tensor& gammaGradient, Tensor& betaGradient, bool trainable, Tensor& inputGradient) const override;
        virtual void Elu(const Tensor& input, float alpha, Tensor& output) const override;
        virtual void EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& inputGradient) const override;
        virtual void Softmax(const Tensor& input, Tensor& output) const override;
        virtual void SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const override;
        virtual void Map(const function<float(float)>& func, const Tensor& t, Tensor& output) const override;
        virtual void Map(const function<float(float, float)>& func, const Tensor& t1, const Tensor& t2, Tensor& output) const override;

    private:
        static cudnnPoolingMode_t GetCudnnPoolType(EPoolingMode type);
        static void GetKernelRunParams(int count, dim3& blocks, dim3& threads);
        static int GetBlocksNum(int count);
        static void CudnnLog(cudnnSeverity_t sev, void *udata, const cudnnDebug_t *dbg, const char *msg);

        static bool s_Initialized;
        static cudaDeviceProp s_CudaDevProp;
        static cublasHandle_t s_CublasHandle;
        static cudnnHandle_t s_CudnnHandle;
    };
}
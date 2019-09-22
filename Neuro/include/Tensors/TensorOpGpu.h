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
        virtual void Sum(const Tensor& input, EAxis axis, Tensor& output) const override;
        virtual void Transpose(const Tensor& input, Tensor& output) const override;
        virtual void Conv2D(const Tensor& input, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& output) const override;
        virtual void Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& inputGradient) const override;
        virtual void Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& kernelsGradient) const override;
        virtual void Pool2D(const Tensor& t, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, Tensor& output) const override;
        virtual void Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, Tensor& inputGradient) const override;
        virtual void BatchNormalization(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, const Tensor& beta, float epsilon, const Tensor& runningMean, const Tensor& runningVar, Tensor& output) const override;
        virtual void BatchNormalizationTrain(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, const Tensor& beta, float momentum, float epsilon, Tensor& runningMean, Tensor& runningVar, Tensor& saveMean, Tensor& saveInvVariance, Tensor& output) const override;
        virtual void BatchNormalizationGradient(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, float epsilon, const Tensor& outputGradient, const Tensor& savedMean, const Tensor& savedInvVariance, Tensor& gammaGradient, Tensor& betaGradient, bool trainable, Tensor& inputGradient) const override;
        virtual void Dropout(const Tensor& input, float prob, Tensor& saveMask, Tensor& output);
        virtual void DropoutGradient(const Tensor& outputGradient, const Tensor& savedMask, Tensor& inputGradient);
        virtual void Sigmoid(const Tensor& input, Tensor& output) const override;
        virtual void SigmoidGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const override;
        virtual void Tanh(const Tensor& input, Tensor& output) const override;
        virtual void TanhGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const override;
        virtual void ReLU(const Tensor& input, Tensor& output) const override;
        virtual void ReLUGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const override;
        virtual void Elu(const Tensor& input, float alpha, Tensor& output) const override;
        virtual void EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& inputGradient) const override;
        virtual void LeakyReLU(const Tensor& input, float alpha, Tensor& output) const override;
        virtual void LeakyReLUGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& inputGradient) const override;
        virtual void Softmax(const Tensor& input, Tensor& output) const override;
        virtual void SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const override;

        virtual void AdamStep(Tensor& parameter, Tensor& gradient, Tensor& mGrad, Tensor& vGrad, float batchSize, float lr, float beta1, float beta2, float epsilon) const override;
        virtual void SgdStep(Tensor& parameter, Tensor& gradient, float batchSize, float lr) const override;

    private:
        void Activation(const cudnnActivationMode_t& activationMode, const Tensor& input, Tensor& output, float coeff) const;
        void ActivationGradient(const cudnnActivationMode_t& activationMode, const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient, float coeff) const;

        void MulGeneric(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const;
        void MulBatched(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const;
        void MulStridedBatched(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const;

        static cudnnPoolingMode_t GetCudnnPoolType(EPoolingMode mode);
        static cudnnBatchNormMode_t GetCudnnBatchNormMode(EBatchNormMode mode);
        static void GetKernelRunParams(int count, dim3& blocks, dim3& threads);
        static int GetBlocksNum(int count);
        static void CudnnLog(cudnnSeverity_t sev, void *udata, const cudnnDebug_t *dbg, const char *msg);

        static bool s_Initialized;
        static cudaDeviceProp s_CudaDevProp;
        static cublasHandle_t s_CublasHandle;
        static cudnnHandle_t s_CudnnHandle;
    };
}
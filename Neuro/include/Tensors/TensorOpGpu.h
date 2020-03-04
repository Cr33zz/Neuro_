#pragma once
#include <cuda.h>
#include <cudnn.h>
#include <cublas.h>
#include <curand.h>

#include "Types.h"
#include "Tensors/TensorOpCpuMt.h"

namespace Neuro
{
    class TensorOpGpu : public TensorOpCpuMt
    {
    public:
        TensorOpGpu();
        virtual EOpMode OpMode() const { return GPU; }

        virtual void Zero(Tensor& input) const override;
        virtual void One(Tensor& input) const override;
        virtual void Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& output) const override;
        virtual void MatMul(const Tensor& t1, bool transposeT1, const Tensor& t2, bool transposeT2, Tensor& output) const override;
        virtual void MatMul(const Tensor& t, bool transpose, Tensor& output) const override;
        virtual void Mul(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& output) const override;
        virtual void Div(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& output) const override;
        virtual void Mul(const Tensor& input, float v, Tensor& output) const override;
        virtual void Scale(Tensor& input, float v) const override;
        virtual void Div(const Tensor& input, float v, Tensor& output) const override;
        virtual void Add(const Tensor& input, float v, Tensor& output) const override;
        virtual void Pow(const Tensor& input, float power, Tensor& output) const override;
        virtual void PowGradient(const Tensor& input, float power, const Tensor& outputGradient, Tensor& inputGradient) const override;
        virtual void Abs(const Tensor& input, Tensor& output) const override;
        virtual void AbsGradient(const Tensor& input, const Tensor& outputGradient, Tensor& inputGradient) const override;
        virtual void Sqrt(const Tensor& input, Tensor& output) const override;
        virtual void Log(const Tensor& input, Tensor& output) const override;
        virtual void Negate(const Tensor& input, Tensor& output) const override;
        virtual void Inverse(float alpha, const Tensor& input, Tensor& output) const override;
        virtual void Clip(const Tensor& input, float min, float max, Tensor& output) const override;
        virtual void ClipGradient(const Tensor& input, float min, float max, const Tensor& outputGradient, Tensor& inputGradient) const override;
        virtual void AbsSum(const Tensor& input, EAxis axis, Tensor& output) const override;
        virtual void Sum(const Tensor& input, EAxis axis, Tensor& output) const override;
        virtual void Mean(const Tensor& input, EAxis axis, Tensor& output) const override;
        virtual void Transpose(const Tensor& input, Tensor& output) const override;
        virtual void Transpose(const Tensor& input, const vector<EAxis>& permutation, Tensor& output) const override;
        virtual void ConstantPad2D(const Tensor& input, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value, Tensor& output) const override;
        virtual void ReflectPad2D(const Tensor& input, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, Tensor& output) const override;
        virtual void Pad2DGradient(const Tensor& gradient, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, Tensor& inputsGradient) const override;
        virtual void Conv2D(const Tensor& input, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& output) const override;
        virtual void Conv2DBiasActivation(const Tensor& input, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, const Tensor& bias, EActivation activation, float activationAlpha, Tensor& output) override;
        virtual void Conv2DBiasGradient(const Tensor& gradient, Tensor& inputsGradient) override;
        virtual void Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& inputGradient) const override;
        virtual void Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& kernelsGradient) const override;
        virtual void Pool2D(const Tensor& t, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& output) const override;
        virtual void Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& inputGradient) const override;
        virtual void UpSample2D(const Tensor& input, uint32_t scaleFactor, Tensor& output) const override;
        virtual void UpSample2DGradient(const Tensor& outputGradient, uint32_t scaleFactor, Tensor& inputGradient) const override;
        virtual void BatchNormalization(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, const Tensor& beta, float epsilon, const Tensor* runningMean, const Tensor* runningVar, Tensor& output) const override;
        virtual void BatchNormalizationTrain(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, const Tensor& beta, float momentum, float epsilon, Tensor* runningMean, Tensor* runningVar, Tensor& saveMean, Tensor& saveInvVariance, Tensor& output) const override;
        virtual void BatchNormalizationGradient(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, float epsilon, const Tensor& outputGradient, const Tensor& savedMean, const Tensor& savedInvVariance, Tensor& gammaGradient, Tensor& betaGradient, bool trainable, Tensor& inputGradient) const override;
        virtual void Dropout(const Tensor& input, float prob, Tensor& saveMask, Tensor& output) const override;
        void DropoutNoRand(const Tensor& input, float prob, Tensor& saveMask, Tensor& output) const;
        virtual void DropoutGradient(const Tensor& outputGradient, float prob, const Tensor& savedMask, Tensor& inputGradient) const override;
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
        virtual void ExtractSubTensor2D(const Tensor& input, uint32_t widthOffset, uint32_t heightOffset, Tensor& output) const override;
        virtual void FuseSubTensor2D(const Tensor& input, uint32_t widthOffset, uint32_t heightOffset, bool add, Tensor& output) const override;
        virtual void AdamStep(Tensor& parameter, const Tensor& gradient, Tensor& mGrad, Tensor& vGrad, float lr, float beta1, float beta2, float epsilon) const override;
        virtual void SgdStep(Tensor& parameter, const Tensor& gradient, float lr) const override;

    private:
        void Reduce(const Tensor& input, cudnnReduceTensorOp_t reductionOp, Tensor& output) const;

        void Activation(const cudnnActivationMode_t& activationMode, const Tensor& input, Tensor& output, float coeff) const;
        void ActivationGradient(const cudnnActivationMode_t& activationMode, const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient, float coeff) const;

        void MatMulGeneric(const Tensor& t1, bool transposeT1, const Tensor& t2, bool transposeT2, Tensor& output) const;
        void MatMulBatched(const Tensor& t1, const Tensor& t2, Tensor& output) const;
        void MatMulStridedBatched(const Tensor& t1, const Tensor& t2, Tensor& output) const;

        static void DeallocateWorkspace(void* ptr);

        static cudnnPoolingMode_t GetCudnnPoolType(EPoolingMode mode);
        static cudnnBatchNormMode_t GetCudnnBatchNormMode(EBatchNormMode mode);
        static cudnnActivationMode_t GetCudnnActivationMode(EActivation mode);
        static void GetKernelRunParamsForSequence(int count, dim3& blocks, dim3& threads, int maxThreads);
        static void GetKernelRunParams(int count, dim3& blocks, dim3& threads, int threadsPerBlock);
        static void CudnnLog(cudnnSeverity_t sev, void *udata, const cudnnDebug_t *dbg, const char *msg);

        static bool s_Initialized;
        static cudaDeviceProp s_CudaDevProp;
        static cublasHandle_t s_CublasHandle;
        static cudnnHandle_t s_CudnnHandle;
        static curandGenerator_t s_CurandGenerator;
    };
}
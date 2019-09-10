#pragma once

#include "Tensors/Tensor.h"

namespace Neuro
{
	class TensorOpCpu
    {
	public:
		virtual ~TensorOpCpu() {}

        virtual void Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& output) const;
        virtual void Sub(const Tensor& t1, const Tensor& t2, Tensor& output) const;
        virtual void Mul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const;
		virtual void MulElem(const Tensor& t1, const Tensor& t2, Tensor& output) const;
        virtual void Sum(const Tensor& input, EAxis axis, int batch, Tensor& output) const;
        virtual void Transpose(const Tensor& input, Tensor& output) const;
        virtual void Conv2D(const Tensor& input, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& output) const;
        virtual void Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& inputGradient) const;
        virtual void Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& kernelsGradient) const;
        virtual void Pool2D(const Tensor& input, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, Tensor& output) const;
        virtual void Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, Tensor& inputGradient) const;
        virtual void UpSample2D(const Tensor& input, uint32_t scaleFactor, Tensor& output) const;
        virtual void UpSample2DGradient(const Tensor& outputGradient, uint32_t scaleFactor, Tensor& inputGradient) const;
        virtual void BatchNormalization(const Tensor& input, const Tensor& gamma, const Tensor& beta, const Tensor& runningMean, const Tensor& runningVar, Tensor& output) const;
        virtual void BatchNormalizationTrain(const Tensor& input, const Tensor& gamma, const Tensor& beta, float momentum, Tensor& runningMean, Tensor& runningVar, Tensor& saveMean, Tensor& saveInvVariance, Tensor& output) const;
        virtual void BatchNormalizationGradient(const Tensor& input, const Tensor& gamma, const Tensor& outputGradient, const Tensor& savedMean, const Tensor& savedInvVariance, Tensor& gammaGradient, Tensor& betaGradient, Tensor& inputGradient) const;
        virtual void Dropout(const Tensor& input, float prob, Tensor& saveMask, Tensor& output);
        virtual void DropoutGradient(const Tensor& outputGradient, const Tensor& savedMask, Tensor& inputGradient);
		virtual void Map(const function<float(float)>& func, const Tensor& t, Tensor& output) const;
        virtual void Map(const function<float(float, float)>& func, const Tensor& t1, const Tensor& t2, Tensor& output) const;
        virtual void Elu(const Tensor& input, float alpha, Tensor& output) const;
        virtual void EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& inputGradient) const;
        virtual void LeakyReLU(const Tensor& input, float alpha, Tensor& output) const;
        virtual void LeakyReLUGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& inputGradient) const;
        virtual void Softmax(const Tensor& input, Tensor& output) const;
        virtual void SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const;
	};
}

#pragma once

#include "Tensors/Tensor.h"

namespace Neuro
{
	class TensorOpCpu
    {
	public:
		virtual ~TensorOpCpu() {}

        virtual void Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& result) const;
        virtual void Sub(const Tensor& t1, const Tensor& t2, Tensor& result) const;
        virtual void Mul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& result) const;
		virtual void MulElem(const Tensor& t1, const Tensor& t2, Tensor& result) const;
        virtual void Transpose(const Tensor& t, Tensor& result) const;
        virtual void Conv2D(const Tensor& t, const Tensor& kernels, int stride, Tensor::EPaddingType padding, Tensor& result) const;
        virtual void Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, int stride, Tensor::EPaddingType padding, Tensor& inputGradients) const;
        virtual void Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, int stride, Tensor::EPaddingType padding, Tensor& kernelsGradient) const;
        virtual void Pool(const Tensor& t, int filterSize, int stride, Tensor::EPoolType type, int paddingX, int paddingY, Tensor& result) const;
        virtual void PoolGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, int filterSize, int stride, Tensor::EPoolType type, int paddingX, int paddingY, Tensor& result) const;
		virtual void Map(const function<float(float)>& func, const Tensor& t, Tensor& result) const;
        virtual void Map(const function<float(float, float)>& func, const Tensor& t1, const Tensor& t2, Tensor& result) const;
        virtual void SumBatches(const Tensor& t, Tensor& result) const;
        virtual void Elu(const Tensor& input, float alpha, Tensor& result) const;
        virtual void EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& result) const;
        virtual void Softmax(const Tensor& input, Tensor& result) const;
        virtual void SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& result) const;
	};
}

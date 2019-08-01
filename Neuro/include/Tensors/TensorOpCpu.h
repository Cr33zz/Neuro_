#pragma once

#include <functional>

namespace Neuro
{
	using namespace std;
	class Tensor;

    class TensorOpCpu
    {
	public:
		virtual ~TensorOpCpu() {}

        virtual void Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& result);

        virtual void Sub(const Tensor& t1, const Tensor& t2, Tensor& result);

        virtual void Mul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& result);        

		virtual void MulElem(const Tensor& t1, const Tensor& t2, Tensor& result);

        virtual void Transpose(Tensor t, Tensor& result);

		virtual void Map(const function<float(float)>& func, const Tensor& t, Tensor& result);

        virtual void Conv2D(const Tensor& t, const Tensor& kernels, int stride, Tensor::EPaddingType padding, Tensor& result);

        virtual void Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, int stride, Tensor::EPaddingType padding, Tensor inputGradients);

        virtual void Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, int stride, Tensor::EPaddingType padding, Tensor kernelsGradient);

        virtual void Conv2DGradient_old(const Tensor& input, const Tensor& kernels, const Tensor& outputGradient, int stride, int paddingX, int paddingY, const Tensor& inputGradient, Tensor kernelsGradient);

        virtual void Pool(const Tensor& t, int filterSize, int stride, Tensor::EPoolType type, int paddingX, int paddingY, Tensor& result);

        virtual void PoolGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, int filterSize, int stride, Tensor::EPoolType type, int paddingX, int paddingY, Tensor& result);

        virtual void Map(const function<float(float, float)>& func, const Tensor& t1, const Tensor& t2, Tensor& result);

        virtual void SumBatches(Tensor t, Tensor& result);

        virtual void Elu(const Tensor& input, float alpha, Tensor& result);

        virtual void EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& result);

        virtual void Softmax(const Tensor& input, Tensor& result);

        virtual void SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& result);
	};
}

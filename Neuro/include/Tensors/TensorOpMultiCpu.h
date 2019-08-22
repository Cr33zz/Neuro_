#pragma once

#include "Tensors/TensorOpCpu.h"

namespace Neuro
{
    class TensorOpMultiCpu : public TensorOpCpu
    {
    public:
        virtual void Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& result) const override;
        virtual void Mul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& result) const override;
        virtual void MulElem(const Tensor& t1, const Tensor& t2, Tensor& result) const override;
        virtual void Transpose(const Tensor& t, Tensor& result) const override;
        virtual void Conv2D(const Tensor& t, const Tensor& kernels, int stride, EPaddingMode padding, Tensor& result) const override;
        virtual void Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, int stride, EPaddingMode padding, Tensor& inputGradients) const override;
        virtual void Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, int stride, EPaddingMode padding, Tensor& kernelsGradient) const override;
        virtual void Pool(const Tensor& t, int filterSize, int stride, EPoolingMode type, int paddingX, int paddingY, Tensor& result) const override;
        virtual void PoolGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, int filterSize, int stride, EPoolingMode type, int paddingX, int paddingY, Tensor& result) const override;
        virtual void Map(const function<float(float)>& func, const Tensor& t, Tensor& result) const override;
        virtual void Map(const function<float(float, float)>& func, const Tensor& t1, const Tensor& t2, Tensor& result) const override;
        virtual void SumBatches(const Tensor& t, Tensor& result) const override;
    };
}

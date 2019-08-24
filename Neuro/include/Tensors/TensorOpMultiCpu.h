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
        virtual void Transpose(const Tensor& input, Tensor& result) const override;
        virtual void Conv2D(const Tensor& input, const Tensor& kernels, int stride, int paddingX, int paddingY, Tensor& result) const override;
        virtual void Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, int stride, int paddingX, int paddingY, Tensor& inputGradients) const override;
        virtual void Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, int stride, int paddingX, int paddingY, Tensor& kernelsGradient) const override;
        virtual void Pool2D(const Tensor& t, int filterSize, int stride, EPoolingMode type, int paddingX, int paddingY, Tensor& result) const override;
        virtual void Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, int filterSize, int stride, EPoolingMode type, int paddingX, int paddingY, Tensor& result) const override;
        virtual void UpSample2D(const Tensor& t, int scaleFactor, Tensor& result) const override;
        virtual void UpSample2DGradient(const Tensor& outputGradient, int scaleFactor, Tensor& result) const override;
        virtual void Map(const function<float(float)>& func, const Tensor& t, Tensor& result) const override;
        virtual void Map(const function<float(float, float)>& func, const Tensor& t1, const Tensor& t2, Tensor& result) const override;
        virtual void SumBatches(const Tensor& t, Tensor& result) const override;
    };
}

#pragma once

#include "Tensors/TensorOpCpu.h"

namespace Neuro
{
    class TensorOpMultiCpu : public TensorOpCpu
    {
    public:
        virtual void Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& output) const override;
        virtual void Mul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const override;
        virtual void MulElem(const Tensor& t1, const Tensor& t2, Tensor& output) const override;
        virtual void Sum(const Tensor& input, EAxis axis, int batch, Tensor& output) const override;
        virtual void Transpose(const Tensor& input, Tensor& output) const override;
        virtual void Conv2D(const Tensor& input, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& output) const override;
        virtual void Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& inputGradient) const override;
        virtual void Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& kernelsGradient) const override;
        virtual void Pool2D(const Tensor& input, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, Tensor& output) const override;
        virtual void Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, Tensor& inputGradient) const override;
        virtual void UpSample2D(const Tensor& t, uint32_t scaleFactor, Tensor& output) const override;
        virtual void UpSample2DGradient(const Tensor& outputGradient, uint32_t scaleFactor, Tensor& inputGradient) const override;
        virtual void Map(const function<float(float)>& func, const Tensor& t, Tensor& output) const override;
        virtual void Map(const function<float(float, float)>& func, const Tensor& t1, const Tensor& t2, Tensor& output) const override;
    };
}

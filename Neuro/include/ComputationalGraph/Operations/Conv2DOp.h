#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class Conv2dOp : public Operation
    {
    public:
        Conv2dOp(TensorLike* x, TensorLike* kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat = NCHW);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        EDataFormat m_DataFormat;
        uint32_t m_Stride;
        uint32_t m_Padding;
    };

    static Operation* conv2d(TensorLike* x, TensorLike* kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat)
    {
        return new Conv2dOp(x, kernels, stride, padding, dataFormat);
    }
}

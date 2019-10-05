#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class Conv2dTransposeOp : public Operation
    {
    public:
        Conv2dTransposeOp(TensorLike* x, TensorLike* kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat = NCHW, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        uint32_t m_OutputDepth;
        uint32_t m_Stride;
        uint32_t m_Padding;
        EDataFormat m_DataFormat;
    };

    static Operation* conv2d_transpose(TensorLike* x, TensorLike* kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat, const string& name = "")
    {
        return new Conv2dTransposeOp(x, kernels, stride, padding, dataFormat, name);
    }
}

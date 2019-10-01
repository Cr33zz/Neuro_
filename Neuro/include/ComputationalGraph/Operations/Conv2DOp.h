#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class Conv2DOp : public Operation
    {
    public:
        Conv2DOp(NodeBase* x, NodeBase* kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat = NCHW);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        EDataFormat m_DataFormat;
        uint32_t m_Stride;
        uint32_t m_Padding;
    };

    static Operation* conv2d(NodeBase* x, NodeBase* kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat)
    {
        return new Conv2DOp(x, kernels, stride, padding, dataFormat);
    }
}

#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class Pool2dOp : public Operation
    {
    public:
        Pool2dOp(TensorLike* x, uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, EDataFormat dataFormat, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        int m_FilterSize;
        int m_Stride;
        int m_Padding;
        EPoolingMode m_Mode;
        EDataFormat m_DataFormat;
    };

    static Operation* pool2d(TensorLike* x, uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, EDataFormat dataFormat, const string& name = "")
    {
        return new Pool2dOp(x, filterSize, stride, padding, mode, dataFormat, name);
    }
}

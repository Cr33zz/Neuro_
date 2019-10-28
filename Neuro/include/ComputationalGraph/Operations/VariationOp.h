#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class VariationOp : public Operation
    {
    public:
        VariationOp(TensorLike* x, EDataFormat dataFormat = NCHW, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        EDataFormat m_DataFormat;
        Tensor m_Kernel;
    };

    static Operation* variation(TensorLike* x, EDataFormat dataFormat = NCHW, const string& name = "")
    {
        return new VariationOp(x, dataFormat, name);
    }
}

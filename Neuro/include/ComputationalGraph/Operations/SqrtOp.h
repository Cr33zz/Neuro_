#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT SqrtOp : public Operation
    {
    public:
        SqrtOp(TensorLike* x, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* sqrt(TensorLike* x, const string& name = "")
    {
        return new SqrtOp(x, name);
    }
}

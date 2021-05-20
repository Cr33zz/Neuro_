#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT ExpOp : public Operation
    {
    public:
        ExpOp(TensorLike* x, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* exp(TensorLike* x, const string& name = "")
    {
        return new ExpOp(x, name);
    }
}

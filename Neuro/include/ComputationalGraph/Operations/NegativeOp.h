#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT NegativeOp : public Operation
    {
    public:
        NegativeOp(TensorLike* x, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* negative(TensorLike* x, const string& name = "")
    {
        return new NegativeOp(x, name);
    }
}

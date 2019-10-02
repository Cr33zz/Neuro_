#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NegativeOp : public Operation
    {
    public:
        NegativeOp(TensorLike* x);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* negative(TensorLike* x)
    {
        return new NegativeOp(x);
    }
}

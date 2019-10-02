#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class SigmoidOp : public Operation
    {
    public:
        SigmoidOp(TensorLike* x);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* sigmoid(TensorLike* x)
    {
        return new SigmoidOp(x);
    }
}

#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class TanHOp : public Operation
    {
    public:
        TanHOp(TensorLike* x);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* tanh(TensorLike* x)
    {
        return new TanHOp(x);
    }
}

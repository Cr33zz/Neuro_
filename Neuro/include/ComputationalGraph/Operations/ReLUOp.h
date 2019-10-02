#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class ReLUOp : public Operation
    {
    public:
        ReLUOp(TensorLike* x);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* relu(TensorLike* x)
    {
        return new ReLUOp(x);
    }
}

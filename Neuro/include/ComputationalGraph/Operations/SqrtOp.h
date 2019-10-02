#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class SqrtOp : public Operation
    {
    public:
        SqrtOp(TensorLike* x);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* sqrt(TensorLike* x)
    {
        return new SqrtOp(x);
    }
}

#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class MultiplyOp : public Operation
    {
    public:
        MultiplyOp(TensorLike* a, TensorLike* b);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* multiply(TensorLike* a, TensorLike* b)
    {
        return new MultiplyOp(a, b);
    }
}

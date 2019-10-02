#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class SubtractOp : public Operation
    {
    public:
        SubtractOp(TensorLike* a, TensorLike* b);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* subtract(TensorLike* a, TensorLike* b)
    {
        return new SubtractOp(a, b);
    }

    static Operation* sub(TensorLike* a, TensorLike* b)
    {
        return subtract(a, b);
    }
}

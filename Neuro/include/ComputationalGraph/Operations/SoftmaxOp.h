#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class SoftmaxOp : public Operation
    {
    public:
        SoftmaxOp(TensorLike* x);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* softmax(TensorLike* x)
    {
        return new SoftmaxOp(x);
    }
}

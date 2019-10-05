#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class MultiplyOp : public Operation
    {
    public:
        MultiplyOp(TensorLike* a, TensorLike* b, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* multiply(TensorLike* a, TensorLike* b, const string& name = "")
    {
        return new MultiplyOp(a, b, name);
    }
}

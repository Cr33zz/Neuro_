#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class SubtractOp : public Operation
    {
    public:
        SubtractOp(TensorLike* a, TensorLike* b, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* subtract(TensorLike* a, TensorLike* b, const string& name = "")
    {
        return new SubtractOp(a, b, name);
    }

    static Operation* sub(TensorLike* a, TensorLike* b, const string& name = "")
    {
        return subtract(a, b, name);
    }
}

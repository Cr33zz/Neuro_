#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class DivideOp : public Operation
    {
    public:
        DivideOp(TensorLike* a, TensorLike* b, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* divide(TensorLike* a, TensorLike* b, const string& name = "")
    {
        return new DivideOp(a, b, name);
    }

    static Operation* div(TensorLike* a, TensorLike* b, const string& name = "")
    {
        return divide(a, b, name);
    }
}

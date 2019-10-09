#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class TransposeOp : public Operation
    {
    public:
        TransposeOp(TensorLike* x, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* transpose(TensorLike* x, const string& name = "")
    {
        return new TransposeOp(x, name);
    }
}

#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class IdentityOp : public Operation
    {
    public:
        IdentityOp(TensorLike* x, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* identity(TensorLike* x, const string& name = "")
    {
        return new IdentityOp(x, name);
    }
}

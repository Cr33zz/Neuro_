#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT TanHOp : public Operation
    {
    public:
        TanHOp(TensorLike* x, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* tanh(TensorLike* x, const string& name = "")
    {
        return new TanHOp(x, name);
    }
}

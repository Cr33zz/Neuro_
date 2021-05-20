#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT AbsOp : public Operation
    {
    public:
        AbsOp(TensorLike* x, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* abs(TensorLike* x, const string& name = "")
    {
        return new AbsOp(x, name);
    }
}

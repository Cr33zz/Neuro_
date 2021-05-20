#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT SigmoidOp : public Operation
    {
    public:
        SigmoidOp(TensorLike* x, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* sigmoid(TensorLike* x, const string& name = "")
    {
        return new SigmoidOp(x, name);
    }
}

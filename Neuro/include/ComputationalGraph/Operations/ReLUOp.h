#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT ReLUOp : public Operation
    {
    public:
        ReLUOp(TensorLike* x, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* relu(TensorLike* x, const string& name = "")
    {
        return new ReLUOp(x, name);
    }
}

#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT SoftmaxOp : public Operation
    {
    public:
        SoftmaxOp(TensorLike* x, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* softmax(TensorLike* x, const string& name = "")
    {
        return new SoftmaxOp(x, name);
    }
}

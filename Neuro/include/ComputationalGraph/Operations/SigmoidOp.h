#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class SigmoidOp : public Operation
    {
    public:
        SigmoidOp(NodeBase* x);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* sigmoid(NodeBase* x)
    {
        return new SigmoidOp(x);
    }
}

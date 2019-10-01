#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class ExpOp : public Operation
    {
    public:
        ExpOp(NodeBase* x);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* exp(NodeBase* x)
    {
        return new ExpOp(x);
    }
}

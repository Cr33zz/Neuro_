#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NegativeOp : public Operation
    {
    public:
        NegativeOp(NodeBase* x);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* negative(NodeBase* x)
    {
        return new NegativeOp(x);
    }
}

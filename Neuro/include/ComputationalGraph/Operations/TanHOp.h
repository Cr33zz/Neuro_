#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class TanHOp : public Operation
    {
    public:
        TanHOp(NodeBase* x);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* tanh(NodeBase* x)
    {
        return new TanHOp(x);
    }
}

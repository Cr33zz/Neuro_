#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class MultiplyOp : public Operation
    {
    public:
        MultiplyOp(NodeBase* a, NodeBase* b);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* multiply(NodeBase* a, NodeBase* b)
    {
        return new MultiplyOp(a, b);
    }
}

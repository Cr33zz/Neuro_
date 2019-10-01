#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class SubtractOp : public Operation
    {
    public:
        SubtractOp(NodeBase* a, NodeBase* b);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* subtract(NodeBase* a, NodeBase* b)
    {
        return new SubtractOp(a, b);
    }
}

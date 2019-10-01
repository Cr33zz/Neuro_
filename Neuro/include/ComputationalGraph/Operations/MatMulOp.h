#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class MatMulOp : public Operation
    {
    public:
        MatMulOp(NodeBase* x1, NodeBase* x2);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* matmul(NodeBase* x1, NodeBase* x2)
    {
        return new MatMulOp(x1, x2);
    }
}

#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    namespace Op
    {
        class MatMul : public Operation
        {
        public:
            MatMul(NodeBase* x1, NodeBase* x2);

        protected:
            virtual void ComputeInternal() override;
            virtual void ComputeGradientInternal(const Tensor& grad) override;
        };
    }

    static Operation* matmul(NodeBase* x1, NodeBase* x2)
    {
        return new Op::MatMul(x1, x2);
    }
}

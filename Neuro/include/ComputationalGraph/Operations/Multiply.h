#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    namespace Op
    {
        class Multiply : public Operation
        {
        public:
            Multiply(NodeBase* a, NodeBase* b);

        protected:
            virtual void ComputeInternal() override;
            virtual void ComputeGradientInternal(const Tensor& grad) override;
        };
    }

    static Operation* multiply(NodeBase* a, NodeBase* b)
    {
        return new Op::Multiply(a, b);
    }
}

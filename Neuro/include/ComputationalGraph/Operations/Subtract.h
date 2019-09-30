#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    namespace Op
    {
        class Subtract : public Operation
        {
        public:
            Subtract(NodeBase* a, NodeBase* b);

        protected:
            virtual void ComputeInternal() override;
            virtual void ComputeGradientInternal(const Tensor& grad) override;
        };
    }

    static Operation* subtract(NodeBase* a, NodeBase* b)
    {
        return new Op::Subtract(a, b);
    }
}

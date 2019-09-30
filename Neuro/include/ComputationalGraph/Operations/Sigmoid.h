#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    namespace Op
    {
        class Sigmoid : public Operation
        {
        public:
            Sigmoid(NodeBase* x);

        protected:
            virtual void ComputeInternal() override;
            virtual void ComputeGradientInternal(const Tensor& grad) override;
        };
    }

    static Operation* sigmoid(NodeBase* x)
    {
        return new Op::Sigmoid(x);
    }
}

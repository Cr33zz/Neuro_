#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    namespace Op
    {
        class Negative : public Operation
        {
        public:
            Negative(NodeBase* x);

        protected:
            virtual void ComputeInternal() override;
            virtual void ComputeGradientInternal(const Tensor& grad) override;
        };
    }

    static Operation* negative(NodeBase* x)
    {
        return new Op::Negative(x);
    }
}

#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    namespace Op
    {
        class Sqrt : public Operation
        {
        public:
            Sqrt(NodeBase* x);

        protected:
            virtual void ComputeInternal() override;
            virtual void ComputeGradientInternal(const Tensor& grad) override;
        };
    }

    static Operation* sqrt(NodeBase* x)
    {
        return new Op::Sqrt(x);
    }
}

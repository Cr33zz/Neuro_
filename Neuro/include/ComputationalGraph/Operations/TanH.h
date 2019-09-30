#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    namespace Op
    {
        class TanH : public Operation
        {
        public:
            TanH(NodeBase* x);

        protected:
            virtual void ComputeInternal() override;
            virtual void ComputeGradientInternal(const Tensor& grad) override;
        };
    }

    static Operation* tanh(NodeBase* x)
    {
        return new Op::TanH(x);
    }
}

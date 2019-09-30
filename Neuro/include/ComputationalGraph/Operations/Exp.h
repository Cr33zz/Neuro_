#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    namespace Op
    {
        class Exp : public Operation
        {
        public:
            Exp(NodeBase* x);

        protected:
            virtual void ComputeInternal() override;
            virtual void ComputeGradientInternal(const Tensor& grad) override;
        };
    }

    static Operation* exp(NodeBase* x)
    {
        return new Op::Exp(x);
    }
}

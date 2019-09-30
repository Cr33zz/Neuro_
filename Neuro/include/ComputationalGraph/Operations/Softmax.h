#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    namespace Op
    {
        class Softmax : public Operation
        {
        public:
            Softmax(NodeBase* x);

        protected:
            virtual void ComputeInternal() override;
            virtual void ComputeGradientInternal(const Tensor& grad) override;
        };
    }

    static Operation* softmax(NodeBase* x)
    {
        return new Op::Softmax(x);
    }
}

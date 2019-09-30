#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    namespace Op
    {
        class Add : public Operation
        {
        public:
            Add(NodeBase* a, NodeBase* b);

        protected:
            virtual void ComputeInternal() override;        
            virtual void ComputeGradientInternal(const Tensor& grad) override;
        };
    }

    static Operation* add(NodeBase* a, NodeBase* b)
    {
        return new Op::Add(a, b);
    }
}

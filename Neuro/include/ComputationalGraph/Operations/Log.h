#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    namespace Op
    {
        class Log : public Operation
        {
        public:
            Log(NodeBase* x);

        protected:
            virtual void ComputeInternal() override;
            virtual void ComputeGradientInternal(const Tensor& grad) override;
        };
    }

    static Operation* log(NodeBase* x)
    {
        return new Op::Log(x);
    }
}

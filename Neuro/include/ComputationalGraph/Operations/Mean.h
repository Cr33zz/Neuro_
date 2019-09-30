#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    namespace Op
    {
        class Mean : public Operation
        {
        public:
            Mean(NodeBase* x, EAxis axis = GlobalAxis);

        protected:
            virtual void ComputeInternal() override;
            virtual void ComputeGradientInternal(const Tensor& grad) override;

        private:
            EAxis m_Axis;
        };
    }

    static Operation* mean(NodeBase* x)
    {
        return new Op::Mean(x);
    }
}

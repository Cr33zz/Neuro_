#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    namespace Op
    {
        class Sum : public Operation
        {
        public:
            Sum(NodeBase* x, EAxis axis = GlobalAxis);

        protected:
            virtual void ComputeInternal() override;
            virtual void ComputeGradientInternal(const Tensor& grad) override;

        private:
            EAxis m_Axis;
        };
    }

    static Operation* sum(NodeBase* x, EAxis axis = GlobalAxis)
    {
        return new Op::Sum(x, axis);
    }
}

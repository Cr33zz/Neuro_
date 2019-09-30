#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    namespace Op
    {
        class Concatenate : public Operation
        {
        public:
            Concatenate(const vector<NodeBase*>& elements, EAxis axis = BatchAxis);

        protected:
            virtual void ComputeInternal() override;
            virtual void ComputeGradientInternal(const Tensor& grad) override;

        private:
            EAxis m_Axis;
        };
    }

    static Operation* concatenate(const vector<NodeBase*>& elements, EAxis axis = BatchAxis)
    {
        return new Op::Concatenate(elements, axis);
    }
}

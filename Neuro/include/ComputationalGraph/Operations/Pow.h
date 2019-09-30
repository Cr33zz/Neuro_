#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    namespace Op
    {
        class Pow : public Operation
        {
        public:
            Pow(NodeBase* x, float p);

        protected:
            virtual void ComputeInternal() override;
            virtual void ComputeGradientInternal(const Tensor& grad) override;

        private:
            float m_Power;
        };
    }

    static Operation* pow(NodeBase* x, float p)
    {
        return new Op::Pow(x, p);
    }
}

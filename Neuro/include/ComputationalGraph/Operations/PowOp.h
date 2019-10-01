#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class PowOp : public Operation
    {
    public:
        PowOp(NodeBase* x, float p);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        float m_Power;
    };
    
    static Operation* pow(NodeBase* x, float p)
    {
        return new PowOp(x, p);
    }
}

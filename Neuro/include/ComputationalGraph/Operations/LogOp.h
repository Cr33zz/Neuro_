#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class LogOp : public Operation
    {
    public:
        LogOp(NodeBase* x);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* log(NodeBase* x)
    {
        return new LogOp(x);
    }
}

#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class AddOp : public Operation
    {
    public:
        AddOp(NodeBase* a, NodeBase* b);

    protected:
        virtual void ComputeInternal() override;        
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* add(NodeBase* a, NodeBase* b)
    {
        return new AddOp(a, b);
    }
}

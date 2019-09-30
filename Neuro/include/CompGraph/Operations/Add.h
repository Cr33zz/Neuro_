#pragma once

#include "CompGraph/Operation.h"

namespace Neuro
{
    class Add : public Operation
    {
    public:
        Add(NodeBase* a, NodeBase* b);

    protected:
        virtual void ComputeInternal() override;        
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* add(NodeBase* a, NodeBase* b)
    {
        return new Add(a, b);
    }
}

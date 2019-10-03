#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class AddOp : public Operation
    {
    public:
        AddOp(TensorLike* a, TensorLike* b, const string& name = "");

    protected:
        virtual void ComputeInternal() override;        
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* add(TensorLike* a, TensorLike* b, const string& name = "")
    {
        return new AddOp(a, b, name);
    }
}

#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class FunctionOp : public Operation
    {
    public:
        FunctionOp(const vector<TensorLike*>& inputs, const vector<TensorLike*>& outputs, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* function(const vector<TensorLike*>& inputs, const vector<TensorLike*>& outputs, const string& name = "")
    {
        return new FunctionOp(inputs, outputs, name);
    }
}

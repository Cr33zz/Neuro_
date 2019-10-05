#pragma once

#include "ComputationalGraph/Operation.h"
#include "ComputationalGraph/Constant.h"

namespace Neuro
{
    class PowOp : public Operation
    {
    public:
        PowOp(TensorLike* x, TensorLike* p, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };
    
    static Operation* pow(TensorLike* x, TensorLike* p, const string& name = "")
    {
        return new PowOp(x, p, name);
    }

    static Operation* square(TensorLike* x, const string& name = "")
    {
        return new PowOp(x, new Constant(2), name.empty() ? "square" : name);
    }
}

#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class AssignOp : public Operation
    {
    public:
        AssignOp(TensorLike* x, TensorLike* val);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* assign(TensorLike* x, TensorLike* val)
    {
        return new AssignOp(x, val);
    }
}

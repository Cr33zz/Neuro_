#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class MatMulOp : public Operation
    {
    public:
        MatMulOp(TensorLike* x1, TensorLike* x2);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* matmul(TensorLike* x1, TensorLike* x2)
    {
        return new MatMulOp(x1, x2);
    }
}

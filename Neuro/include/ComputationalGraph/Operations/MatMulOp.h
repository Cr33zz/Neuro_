#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class MatMulOp : public Operation
    {
    public:
        MatMulOp(TensorLike* x1, TensorLike* x2, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        Tensor m_TransTempA;
        Tensor m_MulTempA;
        Tensor m_TransTempB;
        Tensor m_MulTempB;
    };

    static Operation* matmul(TensorLike* x1, TensorLike* x2, const string& name = "")
    {
        return new MatMulOp(x1, x2, name);
    }
}

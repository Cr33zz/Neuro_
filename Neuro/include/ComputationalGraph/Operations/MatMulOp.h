#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class MatMulOp : public Operation
    {
    public:
        MatMulOp(TensorLike* a, TensorLike* b, const string& name = "");
        
    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        Tensor m_TransTempA;
        Tensor m_MulTempA;
        Tensor m_TransTempB;
        Tensor m_MulTempB;
    };

    class MatMulTransOp : public Operation
    {
    public:
        MatMulTransOp(TensorLike* a, bool transposeA, TensorLike* b, bool transposeB, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        bool m_TransposeA;
        bool m_TransposeB;
    };

    static Operation* matmul(TensorLike* a, TensorLike* b, const string& name = "")
    {
        return new MatMulOp(a, b, name);
    }

    static Operation* matmul(TensorLike* a, bool transposeA, TensorLike* b, bool transposeB, const string& name = "")
    {
        return new MatMulTransOp(a, transposeA, b, transposeB, name);
    }
}

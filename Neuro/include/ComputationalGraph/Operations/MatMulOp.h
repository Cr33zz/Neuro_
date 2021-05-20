#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT MatMulOp : public Operation
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

    class NEURO_DLL_EXPORT MatMulTransOp : public Operation
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

    // Performs A*A' or A'*A
    class NEURO_DLL_EXPORT MatMulSyrkOp : public Operation
    {
    public:
        MatMulSyrkOp(TensorLike* a, bool transpose, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        bool m_Transpose;
    };

    static Operation* matmul(TensorLike* a, TensorLike* b, const string& name = "")
    {
        return new MatMulOp(a, b, name);
    }

    static Operation* matmul(TensorLike* a, bool transposeA, TensorLike* b, bool transposeB, const string& name = "")
    {
        return new MatMulTransOp(a, transposeA, b, transposeB, name);
    }

    static Operation* matmul(TensorLike* a, bool transpose, const string& name = "")
    {
        return new MatMulSyrkOp(a, transpose, name);
    }
}

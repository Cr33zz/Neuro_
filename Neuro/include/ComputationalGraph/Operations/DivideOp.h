#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT DivideOp : public Operation
    {
    public:
        DivideOp(TensorLike* a, TensorLike* b, const string& name = "");
        DivideOp(TensorLike* x, float val, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        float m_Val = 0.f;
    };

    static Operation* divide(TensorLike* a, TensorLike* b, const string& name = "")
    {
        return new DivideOp(a, b, name);
    }

    static Operation* divide(TensorLike* x, float val, const string& name = "")
    {
        return new DivideOp(x, val, name);
    }

    static Operation* div(TensorLike* a, TensorLike* b, const string& name = "")
    {
        return divide(a, b, name);
    }

    static Operation* div(TensorLike* x, float val, const string& name = "")
    {
        return divide(x, val, name);
    }
}

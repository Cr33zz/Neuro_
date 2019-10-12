#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class MultiplyOp : public Operation
    {
    public:
        MultiplyOp(TensorLike* a, TensorLike* b, const string& name = "");
        MultiplyOp(TensorLike* x, float val, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        float m_Val = 0.f;
    };

    static Operation* multiply(TensorLike* a, TensorLike* b, const string& name = "")
    {
        return new MultiplyOp(a, b, name);
    }

    static Operation* multiply(TensorLike* x, float factor, const string& name = "")
    {
        return new MultiplyOp(x, factor, name);
    }
}

#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NormalizeGradientOp : public Operation
    {
    public:
        NormalizeGradientOp(TensorLike* x, size_t order = 1, float scale = 1.f, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        float m_Order;
        float m_Scale;
    };

    static Operation* normalize_gradient(TensorLike* x, size_t order = 1, float scale = 1.f, const string& name = "")
    {
        return new NormalizeGradientOp(x, order, scale, name);
    }
}

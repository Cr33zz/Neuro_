#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NormalizeGradientOp : public Operation
    {
    public:
        NormalizeGradientOp(TensorLike* x, size_t order = 1, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        float m_Order;
    };

    static Operation* normalize_gradient(TensorLike* x, size_t order = 1, const string& name = "")
    {
        return new NormalizeGradientOp(x, order, name);
    }
}

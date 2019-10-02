#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class EluOp : public Operation
    {
    public:
        EluOp(TensorLike* x, float alpha);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        float m_Alpha;
    };

    static Operation* elu(TensorLike* x, float alpha)
    {
        return new EluOp(x, alpha);
    }
}

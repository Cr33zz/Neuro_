#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class DropoutOp : public Operation
    {
    public:
        DropoutOp(TensorLike* x, float prob, TensorLike* training);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        float m_Prob;
        Tensor m_Mask;
    };

    static Operation* dropout(TensorLike* x, float prob, TensorLike* training)
    {
        return new DropoutOp(x, prob, training);
    }
}

#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class MeanOp : public Operation
    {
    public:
        MeanOp(TensorLike* x, EAxis axis = GlobalAxis);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        EAxis m_Axis;
    };

    static Operation* mean(TensorLike* x)
    {
        return new MeanOp(x);
    }
}

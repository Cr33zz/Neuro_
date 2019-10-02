#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class ClipOp : public Operation
    {
    public:
        ClipOp(TensorLike* x, float min, float max);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        float m_Min;
        float m_Max;
    };

    static Operation* clip(TensorLike* x, float min, float max)
    {
        return new ClipOp(x, min, max);
    }
}

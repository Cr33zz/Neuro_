#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class ReshapeOp : public Operation
    {
    public:
        ReshapeOp(TensorLike* x, const Shape& shape);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        Shape m_Shape;
    };

    static Operation* reshape(TensorLike* x, const Shape& shape)
    {
        return new ReshapeOp(x, shape);
    }
}

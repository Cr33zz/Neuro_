#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class ReshapeOp : public Operation
    {
    public:
        ReshapeOp(TensorLike* x, const Shape& shape, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        Shape m_Shape;
    };

    static Operation* reshape(TensorLike* x, const Shape& shape, const string& name = "")
    {
        return new ReshapeOp(x, shape, name);
    }
}

#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class BatchReshapeOp : public Operation
    {
    public:
        BatchReshapeOp(TensorLike* x, const Shape& shape, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        Shape m_Shape;
    };

    static Operation* batch_reshape(TensorLike* x, const Shape& shape, const string& name = "")
    {
        return new BatchReshapeOp(x, shape, name);
    }
}

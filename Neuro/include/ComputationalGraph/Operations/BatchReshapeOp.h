#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT BatchReshapeOp : public Operation
    {
    public:
        BatchReshapeOp(TensorLike* x, const Shape& shape, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
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

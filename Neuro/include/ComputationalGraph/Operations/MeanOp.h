#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class MeanOp : public Operation
    {
    public:
        MeanOp(TensorLike* x, EAxis axis = GlobalAxis, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        EAxis m_Axis;
    };

    static Operation* mean(TensorLike* x, EAxis axis = GlobalAxis, const string& name = "")
    {
        return new MeanOp(x, axis, name);
    }
}

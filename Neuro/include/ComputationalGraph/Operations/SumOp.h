#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class SumOp : public Operation
    {
    public:
        SumOp(TensorLike* x, EAxis axis = GlobalAxis, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        EAxis m_Axis;
    };

    static Operation* sum(TensorLike* x, EAxis axis = GlobalAxis, const string& name = "")
    {
        return new SumOp(x, axis, name);
    }
}

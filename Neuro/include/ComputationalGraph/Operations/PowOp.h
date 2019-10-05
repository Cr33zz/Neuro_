#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class PowOp : public Operation
    {
    public:
        PowOp(TensorLike* x, float p, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        float m_Power;
    };
    
    static Operation* pow(TensorLike* x, float p, const string& name = "")
    {
        return new PowOp(x, p, name);
    }
}

#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT DropoutOp : public Operation
    {
    public:
        DropoutOp(TensorLike* x, float prob, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        float m_Prob;
        Tensor m_Mask;
    };

    static Operation* dropout(TensorLike* x, float prob, const string& name = "")
    {
        return new DropoutOp(x, prob, name);
    }
}

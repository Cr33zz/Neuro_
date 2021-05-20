#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT EluOp : public Operation
    {
    public:
        EluOp(TensorLike* x, float alpha, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        float m_Alpha;
    };

    static Operation* elu(TensorLike* x, float alpha, const string& name = "")
    {
        return new EluOp(x, alpha, name);
    }
}

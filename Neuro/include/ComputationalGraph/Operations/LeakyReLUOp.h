#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT LeakyReLUOp : public Operation
    {
    public:
        LeakyReLUOp(TensorLike* x, float alpha, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        float m_Alpha;
    };

    static Operation* leaky_relu(TensorLike* x, float alpha, const string& name = "")
    {
        return new LeakyReLUOp(x, alpha, name);
    }
}

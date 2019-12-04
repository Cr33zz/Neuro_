#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class Conv2dBiasActivationOp : public Operation
    {
    public:
        Conv2dBiasActivationOp(TensorLike* x, TensorLike* kernels, uint32_t stride, uint32_t padding, TensorLike* bias, EActivation activation, float activationAlpha, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        uint32_t m_Stride;
        uint32_t m_Padding;
        EActivation m_Activation;
        float m_ActivationAlpha;

        Tensor m_OutputGradTemp;
    };

    static Operation* conv2d_bias_activation(TensorLike* x, TensorLike* kernels, uint32_t stride, uint32_t padding, TensorLike* bias, EActivation activation, float activationAlpha, const string& name = "")
    {
        return new Conv2dBiasActivationOp(x, kernels, stride, padding, bias, activation, activationAlpha, name);
    }
}

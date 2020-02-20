#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class InstanceNormalizeOp : public Operation
    {
    public:
        InstanceNormalizeOp(TensorLike* x, TensorLike* gamma, TensorLike* beta, float epsilon, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        float m_Epsilon;

        // Used as cache between forward and backward steps
        Tensor m_SaveMean;
        // Used as cache between forward and backward steps
        Tensor m_SaveInvVar;
    };

    static Operation* instance_norm(TensorLike* x, TensorLike* gamma, TensorLike* beta, float epsilon, const string& name = "")
    {
        return new InstanceNormalizeOp(x, gamma, beta, epsilon, name);
    }
}

#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class BatchNormalizeOp : public Operation
    {
    public:
        BatchNormalizeOp(TensorLike* x, TensorLike* gamma, TensorLike* beta, TensorLike* runningMean, TensorLike* runningVar, float momentum, float epsilon, TensorLike* training, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

        virtual bool ForceAllocInputGradNode(size_t index) const override;

    private:
        float m_Momentum;
        float m_Epsilon;

        // Used as cache between forward and backward steps
        Tensor m_SaveMean;
        // Used as cache between forward and backward steps
        Tensor m_SaveInvVar;
    };

    static Operation* batch_norm(TensorLike* x, TensorLike* gamma, TensorLike* beta, TensorLike* runningMean, TensorLike* runningVar, float momentum, float epsilon, TensorLike* training, const string& name = "")
    {
        return new BatchNormalizeOp(x, gamma, beta, runningMean, runningVar, momentum, epsilon, training, name);
    }
}

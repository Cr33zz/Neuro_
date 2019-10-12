#pragma once

#include "ComputationalGraph/Operations/BatchNormalizeOp.h"

namespace Neuro
{
    class InstanceNormalizeOp : public BatchNormalizeOp
    {
    public:
        InstanceNormalizeOp(TensorLike* x, TensorLike* gamma, TensorLike* beta, TensorLike* runningMean, TensorLike* runningVar, float momentum, float epsilon, TensorLike* training, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* instance_norm(TensorLike* x, TensorLike* gamma, TensorLike* beta, TensorLike* runningMean, TensorLike* runningVar, float momentum, float epsilon, TensorLike* training, const string& name = "")
    {
        return new InstanceNormalizeOp(x, gamma, beta, runningMean, runningVar, momentum, epsilon, training, name);
    }
}

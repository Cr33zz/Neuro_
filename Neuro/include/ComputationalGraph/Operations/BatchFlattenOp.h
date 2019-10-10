#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class BatchFlattenOp : public Operation
    {
    public:
        BatchFlattenOp(TensorLike* x, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* batch_flatten(TensorLike* x, const string& name = "")
    {
        return new BatchFlattenOp(x, name);
    }
}

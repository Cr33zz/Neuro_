#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT BatchFlattenOp : public Operation
    {
    public:
        BatchFlattenOp(TensorLike* x, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* batch_flatten(TensorLike* x, const string& name = "")
    {
        return new BatchFlattenOp(x, name);
    }
}

#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT AssignOp : public Operation
    {
    public:
        AssignOp(TensorLike* x, TensorLike* val, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override { assert(false); }
    };

    static Operation* assign(TensorLike* x, TensorLike* val, const string& name = "")
    {
        return new AssignOp(x, val, name);
    }
}

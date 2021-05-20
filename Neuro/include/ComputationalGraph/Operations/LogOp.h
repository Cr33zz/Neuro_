#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT LogOp : public Operation
    {
    public:
        LogOp(TensorLike* x, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* log(TensorLike* x, const string& name = "")
    {
        return new LogOp(x, name);
    }
}

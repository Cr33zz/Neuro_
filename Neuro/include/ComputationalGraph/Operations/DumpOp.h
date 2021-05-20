#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT DumpOp : public Operation
    {
    public:
        DumpOp(TensorLike* x, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* dump(TensorLike* x, const string& name = "")
    {
        return new DumpOp(x, name);
    }
}

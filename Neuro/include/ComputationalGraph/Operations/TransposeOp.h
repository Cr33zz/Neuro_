#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class TransposeOp : public Operation
    {
    public:
        TransposeOp(TensorLike* x, const vector<EAxis>& axes, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        vector<EAxis> m_Permutation;
    };

    static Operation* transpose(TensorLike* x, const vector<EAxis>& axes = { _1Axis, _0Axis }, const string& name = "")
    {
        return new TransposeOp(x, axes, name);
    }
}

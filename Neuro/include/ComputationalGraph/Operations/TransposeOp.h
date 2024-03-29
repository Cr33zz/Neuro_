#pragma once

#include "ComputationalGraph/Operation.h"

#pragma warning(push)
#pragma warning(disable:4251)

namespace Neuro
{
    class NEURO_DLL_EXPORT TransposeOp : public Operation
    {
    public:
        TransposeOp(TensorLike* x, const vector<EAxis>& axes, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        vector<EAxis> m_Permutation;
        vector<EAxis> m_InvPermutation;
    };

    static Operation* transpose(TensorLike* x, const vector<EAxis>& axes = { _1Axis, _0Axis }, const string& name = "")
    {
        return new TransposeOp(x, axes, name);
    }
}

#pragma warning(pop)
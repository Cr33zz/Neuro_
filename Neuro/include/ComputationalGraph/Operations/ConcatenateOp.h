#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class ConcatenateOp : public Operation
    {
    public:
        ConcatenateOp(const vector<TensorLike*>& xs, EAxis axis = DepthAxis, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

        virtual bool ForceAllocInputGradNode(size_t index) const override;

    private:
        EAxis m_Axis;
    };

    static Operation* concat(const vector<TensorLike*>& xs, EAxis axis = DepthAxis, const string& name = "")
    {
        return new ConcatenateOp(xs, axis, name);
    }

    static Operation* concatenate(const vector<TensorLike*>& xs, EAxis axis = DepthAxis, const string& name = "")
    {
        return new ConcatenateOp(xs, axis, name);
    }
}

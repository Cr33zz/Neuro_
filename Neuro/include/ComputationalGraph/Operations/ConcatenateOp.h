#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class ConcatenateOp : public Operation
    {
    public:
        ConcatenateOp(const vector<TensorLike*>& elements, EAxis axis = BatchAxis, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        EAxis m_Axis;
    };

    static Operation* concat(const vector<TensorLike*>& elements, EAxis axis = BatchAxis, const string& name = "")
    {
        return new ConcatenateOp(elements, axis, name);
    }
}

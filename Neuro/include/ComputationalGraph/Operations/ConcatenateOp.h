#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class ConcatenateOp : public Operation
    {
    public:
        ConcatenateOp(const vector<NodeBase*>& elements, EAxis axis = BatchAxis);

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        EAxis m_Axis;
    };

    static Operation* concatenate(const vector<NodeBase*>& elements, EAxis axis = BatchAxis)
    {
        return new ConcatenateOp(elements, axis);
    }
}

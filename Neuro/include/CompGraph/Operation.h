#pragma once

#include "CompGraph/NodeBase.h"

namespace Neuro
{
    class Tensor;

    class Operation : public NodeBase
    {
    public:
        virtual bool IsOp() const override { return true; }

        const vector<Tensor*>& Inputs() const { return m_Inputs; }

        const Tensor& Compute(const vector<Tensor*>& inputs);
        const vector<Tensor*>& ComputeGradient(const Tensor& grad);

    protected:
        Operation(const vector<NodeBase*>& inputNodes);

        virtual void ComputeInternal() = 0;
        virtual void ComputeGradientInternal(const Tensor& grad) = 0;

        vector<Tensor*> m_Inputs;
        vector<Tensor> m_InputsGrads;
        vector<Tensor*> m_InputsGradsPtrs; // for performance
    };
}

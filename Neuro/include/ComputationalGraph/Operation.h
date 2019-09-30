#pragma once

#include "ComputationalGraph/NodeBase.h"

namespace Neuro
{
    class Tensor;

    class Operation : public NodeBase
    {
    public:
        virtual bool IsOp() const override { return true; }

        vector<const Tensor*> GatherInputs() const;

        const Tensor& Compute(const vector<const Tensor*>& inputs);
        const vector<Tensor*>& ComputeGradient(const Tensor& grad);

    protected:
        Operation(const vector<NodeBase*>& inputNodes);

        virtual void ComputeInternal() = 0;
        virtual void ComputeGradientInternal(const Tensor& grad) = 0;

        vector<const Tensor*> m_Inputs;
        vector<Tensor> m_InputsGrads;
        vector<Tensor*> m_InputsGradsPtrs; // for performance
    };
}

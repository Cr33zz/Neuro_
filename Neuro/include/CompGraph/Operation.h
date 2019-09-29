#pragma once

#include "CompGraph/NodeBase.h"

namespace Neuro
{
    class Tensor;

    class Operation : public NodeBase
    {
    public:
        const Tensor& Compute(const vector<Tensor*>& inputs)
        {
            m_Inputs = inputs;
            return ComputeInternal();
        }

        const vector<Tensor*> ComputeGradient(Tensor grad)
        {

        }

    protected:
        Operation(const vector<NodeBase*>& inputNodes)
        {
            m_InputNodes = inputNodes;

            for (auto inputNode : inputNodes)
                inputNode->m_Consumers.push_back(this);

            Graph::Default()->m_Operations.push_back(this);
        }

        virtual const Tensor& ComputeInternal() = 0;
        virtual vector<Tensor*> ComputeGradientInternal(const Tensor& grad) = 0;

        vector<Tensor*> m_Inputs;
        vector<Tensor> m_InputsGrads;
    };
}

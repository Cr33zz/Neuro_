#include "CompGraph/Operation.h"
#include "CompGraph/Graph.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    const Tensor& Operation::Compute(const vector<Tensor*>& inputs)
    {
        m_Inputs = inputs;
        ComputeInternal();
        return m_Output;
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<Tensor*>& Operation::ComputeGradient(const Tensor& grad)
    {
        m_InputsGradsPtrs.resize(m_Inputs.size());
        ComputeGradientInternal(grad);
        return m_InputsGradsPtrs;
    }

    //////////////////////////////////////////////////////////////////////////
    Operation::Operation(const vector<NodeBase*>& inputNodes)
    {
        m_InputNodes = inputNodes;

        for (auto inputNode : inputNodes)
            inputNode->m_Consumers.push_back(this);

        Graph::Default()->Operations.push_back(this);
    }
}

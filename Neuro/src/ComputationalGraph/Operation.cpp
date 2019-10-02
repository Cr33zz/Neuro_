#include "ComputationalGraph/Operation.h"
#include "ComputationalGraph/Graph.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Operation::Operation(const vector<TensorLike*>& inputNodes, const string& name)
        : TensorLike(name)
    {
        m_InputNodes = inputNodes;

        for (auto inputNode : inputNodes)
            inputNode->m_Consumers.push_back(this);

        //Graph::Default()->Operations.push_back(this);
    }

    //////////////////////////////////////////////////////////////////////////
    vector<const Tensor*> Operation::GatherInputs() const
    {
        vector<const Tensor*> inputTensors;
        for (auto inputNode : m_InputNodes)
            inputTensors.push_back(inputNode->OutputPtr());
        return inputTensors;
    }

    //////////////////////////////////////////////////////////////////////////
    const Tensor& Operation::Compute(const vector<const Tensor*>& inputs)
    {
        m_Inputs = inputs;
        ComputeInternal();
        return m_Output;
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<Tensor*>& Operation::ComputeGradient(const Tensor& grad)
    {
        m_InputsGradsPtrs.resize(m_Inputs.size());
        m_InputsGrads.resize(m_Inputs.size());
        for (size_t i = 0; i < m_InputsGrads.size(); ++i)
        {
            m_InputsGrads[i].Resize(m_Inputs[i]->GetShape());
            m_InputsGradsPtrs[i] = &m_InputsGrads[i];
        }
        ComputeGradientInternal(grad);
        return m_InputsGradsPtrs;
    }
}

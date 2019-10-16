#include "ComputationalGraph/Operation.h"
#include "ComputationalGraph/Graph.h"
#include "Tensors/Tensor.h"
#include "Tensors/Cuda/CudaDeviceVariable.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Operation::Operation(const vector<TensorLike*>& inputNodes, const string& name)
        : TensorLike(name)
    {
        m_InputNodes = inputNodes;

        for (auto inputNode : inputNodes)
            inputNode->m_Consumers.push_back(this);

        m_InputsGradsPtrs.resize(m_InputNodes.size());
        m_InputsGrads.resize(m_InputNodes.size());
        for (size_t i = 0; i < m_InputsGrads.size(); ++i)
        {
            m_InputsGrads[i].Resize(m_InputNodes[i]->GetShape());
            m_InputsGrads[i].Name(m_Name + "/inputGrad" + to_string(i));
            m_InputsGradsPtrs[i] = &m_InputsGrads[i];
        }

        m_Output.SetOffloadMode(Offload_Enabled);

        Graph::Default()->AddOperation(this);
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
        m_OutputConsumedCount = 0;
        if (m_Output.TryDeviceAllocate())
            m_Output.OverrideDevice();
        m_Inputs = inputs;

        ComputeInternal();

        m_LastComputeStep = m_Graph->CurrentStep();
        for (auto inputNode : m_InputNodes)
            inputNode->OutputConsumed();
        m_Output.Offload(); // at this point output won't change so start offloading it, it will be released when all consumers used it
        return m_Output;
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<Tensor*>& Operation::ComputeGradient(const Tensor& grad)
    {
        for (size_t i = 0; i < m_InputsGrads.size(); ++i)
            m_InputsGrads[i].ResizeBatch(m_Inputs[i]->GetShape().Batch());

        ComputeGradientInternal(grad);

        m_Output.TryDeviceRelease(); // output is no longer needed, we've already used it to compute input gradients
        m_OutputGrad.TryDeviceRelease(); // output grad is no longer needed, we've already used it to compute input gradients        
        return m_InputsGradsPtrs;
    }

    //////////////////////////////////////////////////////////////////////////
    void Operation::OutputConsumed()
    {
        ++m_OutputConsumedCount;
        if (m_OutputConsumedCount == m_Consumers.size())
            m_Output.TryDeviceRelease();
    }

    //////////////////////////////////////////////////////////////////////////
    void Operation::InputGradConsumed(TensorLike* inputNode)
    {
        for (size_t i = 0; i < m_InputNodes.size(); ++i)
        {
            if (m_InputNodes[i] == inputNode)
            {
                m_InputsGrads[i].TryDeviceRelease();
                return;
            }
        }
        
        NEURO_ASSERT(false, "Unknown node consumed our input O_o");
    }
}

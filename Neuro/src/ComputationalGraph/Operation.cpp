#include "ComputationalGraph/Operation.h"
#include "ComputationalGraph/Graph.h"
#include "Tensors/Tensor.h"
#include "Tools.h"
#include "Debug.h"

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

        m_Output.SetStorageType(ST_DeviceRefCounted|ST_RefCounted|ST_Offloadable);

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
        if (m_Output.TryDeviceAllocate())
            m_Output.OverrideDevice();
        m_Output.ResetDeviceRef(m_Consumers.size());
        m_Output.IncRef();
        m_Inputs = inputs;
        m_InputsManuallyConsumed = false;

        ComputeInternal();

        m_LastComputeStep = m_Graph->CurrentStep();
        if (!m_InputsManuallyConsumed)
        {
            for (auto inputNode : m_InputNodes)
                inputNode->OutputConsumed();
        }

        if (Debug::ShouldLogOutput(Name()))
            m_Output.DebugDumpValues(Replace(Name() + "_step" + to_string(m_LastComputeStep) + ".log", "/", "_"));

        m_Output.Offload(); // at this point output won't change so start offloading it, it will be released when all consumers used it
        return m_Output;
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<Tensor*>& Operation::ComputeGradient(const Tensor& grad)
    {
        for (size_t i = 0; i < m_InputsGrads.size(); ++i)
        {
            m_InputsGrads[i].ResizeBatch(m_Inputs[i]->GetShape().Batch());
            if (m_InputsGrads[i].TryDeviceAllocate())
                m_InputsGrads[i].OverrideDevice();
        }

        ComputeGradientInternal(grad);

        if (Debug::ShouldLogGrad(Name()))
        {
            for (size_t i = 0; i < m_InputsGrads.size(); ++i)
                m_InputsGrads[i].DebugDumpValues(Replace(Name() + "_grad" + to_string(i) + "_step" + to_string(m_LastComputeStep) + ".log", "/", "_"));
        }

        m_Output.DecRef(); // output is no longer needed, we've already used it to compute input gradients
        m_OutputGrad.ReleaseData(); // output grad is no longer needed, we've already used it to compute input gradients
        return m_InputsGradsPtrs;
    }

    //////////////////////////////////////////////////////////////////////////
    void Operation::OutputConsumed()
    {
        m_Output.DecDeviceRef();
    }

    //////////////////////////////////////////////////////////////////////////
    void Operation::InputGradConsumed(TensorLike* inputNode)
    {
        for (size_t i = 0; i < m_InputNodes.size(); ++i)
        {
            if (m_InputNodes[i] == inputNode)
            {
                m_InputsGrads[i].ReleaseData();
                return;
            }
        }
        
        NEURO_ASSERT(false, "Unknown node consumed our input O_o");
    }
}

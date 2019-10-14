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
        m_Output.TryDeviceAllocate();
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
        m_InputGradsConsumedCount = 0;
        for (size_t i = 0; i < m_InputsGrads.size(); ++i)
        {
            m_InputsGrads[i].ResizeBatch(m_Inputs[i]->GetShape().Batch());
            m_InputsGrads[i].TryDeviceAllocate();
        }
        for (auto inputNode : m_InputNodes)
            inputNode->Output().Prefetch();
        ComputeGradientInternal(grad);
        for (auto consumerNode : m_Consumers)
            consumerNode->InputGradConsumed();
        return m_InputsGradsPtrs;
    }

    //////////////////////////////////////////////////////////////////////////
    void Operation::OutputConsumed()
    {
        ++m_OutputConsumedCount;
        if (m_OutputConsumedCount == m_Consumers.size())
            m_Output.DeviceRelease();
    }

    //////////////////////////////////////////////////////////////////////////
    void Operation::InputGradConsumed()
    {
        ++m_InputGradsConsumedCount;
        if (m_InputGradsConsumedCount == m_InputNodes.size())
        {
            for (auto inputGrad : m_InputsGrads)
                inputGrad.DeviceRelease();
        }
    }
}

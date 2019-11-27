#include "ComputationalGraph/Operations/BatchNormalizeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    BatchNormalizeOp::BatchNormalizeOp(TensorLike* x, TensorLike* gamma, TensorLike* beta, TensorLike* runningMean, TensorLike* runningVar, float momentum, float epsilon, const string& name)
        : Operation({ x, gamma, beta, runningMean, runningVar }, name.empty() ? "batch_normalize" : name), m_Epsilon(epsilon), m_Momentum(momentum)
    {
        m_Output.Resize(x->GetShape());
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalizeOp::ComputeInternal()
    {
        auto& x = *m_Inputs[0];
        auto& gamma = *m_Inputs[1];
        auto& beta = *m_Inputs[2];
        auto& runningMean = m_InputNodes[3]->Output();
        auto& runningVar = m_InputNodes[4]->Output();

        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_SaveMean.Resize(gamma.GetShape());
        m_SaveInvVar.Resize(gamma.GetShape());

        if (m_Training)
            m_Inputs[0]->BatchNormTrain(gamma, beta, 1.f - m_Momentum, m_Epsilon, &runningMean, &runningVar, m_SaveMean, m_SaveInvVar, m_Output);
        else
            m_Inputs[0]->BatchNorm(gamma, beta, m_Epsilon, &runningMean, &runningVar, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalizeOp::ComputeGradientInternal(const Tensor& grad)
    {
        auto& x = *m_Inputs[0];
        auto& gamma = *m_Inputs[1];
        auto& beta = *m_Inputs[2];

        if (m_InputNodes[0]->CareAboutGradient() || m_InputNodes[1]->CareAboutGradient() || m_InputNodes[2]->CareAboutGradient())
            grad.BatchNormGradient(x, gamma, m_Epsilon, grad, m_SaveMean, m_SaveInvVar, m_InputsGrads[1], m_InputsGrads[2], true, m_InputsGrads[0]);
    }

    //////////////////////////////////////////////////////////////////////////
    bool BatchNormalizeOp::ForceAllocInputGradNode(size_t index) const
    {
        // we cannot compute input gradient per input separately so if any input needs gradient we have to allocate them all
        return m_InputNodes[0]->CareAboutGradient() || m_InputNodes[1]->CareAboutGradient() || m_InputNodes[2]->CareAboutGradient();
    }

}
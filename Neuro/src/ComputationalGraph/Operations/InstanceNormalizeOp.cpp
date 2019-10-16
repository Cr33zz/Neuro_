#include "ComputationalGraph/Operations/InstanceNormalizeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    InstanceNormalizeOp::InstanceNormalizeOp(TensorLike* x, TensorLike* gamma, TensorLike* beta, TensorLike* runningMean, TensorLike* runningVar, float momentum, float epsilon, TensorLike* training, const string& name)
        : BatchNormalizeOp(x, gamma, beta, runningMean, runningVar, momentum, epsilon, training, name.empty() ? "instance_normalize" : name)
    {
        m_Output.Resize(x->GetShape());
    }

    //////////////////////////////////////////////////////////////////////////
    void InstanceNormalizeOp::ComputeInternal()
    {
        auto& x = *m_Inputs[0];
        auto& gamma = *m_Inputs[1];
        auto& beta = *m_Inputs[2];
        auto& runningMean = m_InputNodes[3]->Output();
        auto& runningVar = m_InputNodes[4]->Output();
        bool training = (*m_Inputs[5])(0) != 0;

        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_SaveMean.Resize(gamma.GetShape());
        m_SaveInvVar.Resize(gamma.GetShape());

        if (training)
            m_Inputs[0]->InstanceNormTrain(gamma, beta, m_Momentum, m_Epsilon, runningMean, runningVar, m_SaveMean, m_SaveInvVar, m_Output);
        else
            m_Inputs[0]->InstanceNorm(gamma, beta, m_Epsilon, runningMean, runningVar, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void InstanceNormalizeOp::ComputeGradientInternal(const Tensor& grad)
    {
        auto& x = *m_Inputs[0];
        auto& gamma = *m_Inputs[1];
        auto& beta = *m_Inputs[2];

        if (m_InputNodes[0]->CareAboutGradient() || m_InputNodes[1]->CareAboutGradient() || m_InputNodes[2]->CareAboutGradient())
            grad.InstanceNormGradient(x, gamma, m_Epsilon, grad, m_SaveMean, m_SaveInvVar, m_InputsGrads[1], m_InputsGrads[2], true, m_InputsGrads[0]);
    }
}
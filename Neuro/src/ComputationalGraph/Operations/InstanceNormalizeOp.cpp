#include "ComputationalGraph/Operations/InstanceNormalizeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    InstanceNormalizeOp::InstanceNormalizeOp(TensorLike* x, TensorLike* gamma, TensorLike* beta, float epsilon, const string& name)
        : Operation({ x, gamma, beta }, name.empty() ? "instance_normalize" : name), m_Epsilon(epsilon)
    {
        m_Output.Resize(x->GetShape());
    }

    //////////////////////////////////////////////////////////////////////////
    void InstanceNormalizeOp::ComputeInternal()
    {
        auto& x = *m_Inputs[0];
        auto& gamma = *m_Inputs[1];
        auto& beta = *m_Inputs[2];
        
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_SaveMean.Resize(Shape(1, 1, x.GetShape().Depth(), x.GetShape().Batch()));
        m_SaveInvVar.Resize(m_SaveMean.GetShape());

        if (m_Training)
            m_Inputs[0]->InstanceNormTrain(gamma, beta, m_Epsilon, m_SaveMean, m_SaveInvVar, m_Output);
        else
            m_Inputs[0]->InstanceNorm(gamma, beta, m_Epsilon, m_Output);
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
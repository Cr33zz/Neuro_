#include "ComputationalGraph/Operations/SigmoidOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    SigmoidOp::SigmoidOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "sigmoid" : name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void SigmoidOp::ComputeInternal()
    {
        m_Output.Resize(m_Inputs[0]->GetShape());
        m_Inputs[0]->Sigmoid(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void SigmoidOp::ComputeGradientInternal(const Tensor& grad)
    {
        m_Output.SigmoidGradient(m_Output, grad, m_InputsGrads[0]);
    }
}
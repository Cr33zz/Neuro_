#include "ComputationalGraph/Operations/SoftmaxOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    SoftmaxOp::SoftmaxOp(NodeBase* x)
        : Operation({x})
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void SoftmaxOp::ComputeInternal()
    {
        m_Output.Resize(m_Inputs[0]->GetShape());
        m_Inputs[0]->Softmax(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void SoftmaxOp::ComputeGradientInternal(const Tensor& grad)
    {
        m_Output.SoftmaxGradient(m_Output, grad, m_InputsGrads[0]);
    }
}
#include "ComputationalGraph/Operations/Softmax.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Op::Softmax::Softmax(NodeBase* x)
        : Operation({x})
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Softmax::ComputeInternal()
    {
        m_Output.Resize(m_Inputs[0]->GetShape());
        m_Inputs[0]->Softmax(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Softmax::ComputeGradientInternal(const Tensor& grad)
    {
        m_Output.SoftmaxGradient(m_Output, grad, m_InputsGrads[0]);
    }
}
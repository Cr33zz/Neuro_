#include "ComputationalGraph/Operations/Sigmoid.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Op::Sigmoid::Sigmoid(NodeBase* x)
        : Operation({x})
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Sigmoid::ComputeInternal()
    {
        m_Inputs[0]->Sigmoid(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Sigmoid::ComputeGradientInternal(const Tensor& grad)
    {
        m_Output.SigmoidGradient(m_Output, grad, m_InputsGrads[0]);
    }
}
#include "ComputationalGraph/Operations/NegativeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    NegativeOp::NegativeOp(NodeBase* x)
        : Operation({x})
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void NegativeOp::ComputeInternal()
    {
        m_Output.Resize(m_Inputs[0]->GetShape());
        m_Inputs[0]->Map([](float x) {return -x; }, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void NegativeOp::ComputeGradientInternal(const Tensor& grad)
    {
        //in_grad = -grad
        grad.Map([](float g) {return -g; }, m_InputsGrads[0]);
    }
}
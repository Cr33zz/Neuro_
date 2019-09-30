#include "ComputationalGraph/Operations/Negative.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Op::Negative::Negative(NodeBase* x)
        : Operation({x})
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Negative::ComputeInternal()
    {
        m_Inputs[0]->Map([](float x) {return -x; }, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Negative::ComputeGradientInternal(const Tensor& grad)
    {
        //in_grad = -grad
        grad.Map([](float g) {return -g; }, m_InputsGrads[0]);
    }
}
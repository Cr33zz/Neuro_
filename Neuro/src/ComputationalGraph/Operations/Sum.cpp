#include "ComputationalGraph/Operations/Sum.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Op::Sum::Sum(NodeBase* x, EAxis axis)
        : Operation({ x })
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Sum::ComputeInternal()
    {
        m_Inputs[0]->Sum(m_Axis, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Sum::ComputeGradientInternal(const Tensor& grad)
    {
        m_InputsGrads[0].FillWithValue(1);
        m_InputsGrads[0].MulElem(grad, m_InputsGrads[0]);
    }
}
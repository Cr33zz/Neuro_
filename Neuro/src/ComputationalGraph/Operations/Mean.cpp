#include "ComputationalGraph/Operations/Mean.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Op::Mean::Mean(NodeBase* x, EAxis axis)
        : Operation({x})
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Mean::ComputeInternal()
    {
        m_Inputs[0]->Mean(m_Axis, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Mean::ComputeGradientInternal(const Tensor& grad)
    {
        float n = (float)m_Inputs[0]->Length();
        if (m_Axis != GlobalAxis)
            n = (float)m_Inputs[0]->Stride(m_Axis);

        m_InputsGrads[0].FillWithValue(1);
        m_InputsGrads[0].MulElem(grad.Div(n), m_InputsGrads[0]);
    }
}
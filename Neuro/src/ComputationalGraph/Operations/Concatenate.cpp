#include "ComputationalGraph/Operations/Concatenate.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Op::Concatenate::Concatenate(const vector<NodeBase*>& elements, EAxis axis)
        : Operation(elements), m_Axis(axis)
    {
        assert(axis == BatchAxis);
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Concatenate::ComputeInternal()
    {
        if (m_Axis == BatchAxis)
            m_Output.Resize(Shape::From(m_Inputs[0]->GetShape(), m_Inputs[0]->Len(3) * (uint32_t)m_Inputs.size()));

        Tensor::Concat(m_Axis, m_Inputs, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Concatenate::ComputeGradientInternal(const Tensor& grad)
    {
        grad.Split(m_Axis, m_InputsGradsPtrs);
    }
}
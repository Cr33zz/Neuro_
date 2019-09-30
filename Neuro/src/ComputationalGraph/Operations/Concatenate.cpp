#include "ComputationalGraph/Operations/Concatenate.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Op::Concatenate::Concatenate(const vector<NodeBase*>& elements, EAxis axis)
        : Operation(elements), m_Axis(axis)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Concatenate::ComputeInternal()
    {
        Tensor::Concat(m_Axis, m_Inputs, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Concatenate::ComputeGradientInternal(const Tensor& grad)
    {
        grad.Split(m_Axis, m_InputsGradsPtrs);
    }
}
#include "ComputationalGraph/Operations/MatMul.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Op::MatMul::MatMul(NodeBase* a, NodeBase* b)
        : Operation({a, b})
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::MatMul::ComputeInternal()
    {
        m_Inputs[0]->Mul(*m_Inputs[1], m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::MatMul::ComputeGradientInternal(const Tensor& grad)
    {
        auto& a = *m_Inputs[0];
        auto& b = *m_Inputs[1];

        auto gradWrtA = grad.Mul(b.Transposed());

        if (m_InputsGrads[0].Len(3) != a.Len(3))
            gradWrtA.Sum(BatchAxis, m_InputsGrads[0]);
        else
            gradWrtA.CopyTo(m_InputsGrads[0]);

        auto gradWrtB = a.Transposed().Mul(grad);

        if (m_InputsGrads[1].Len(3) != b.Len(3))
            gradWrtB.Sum(BatchAxis, m_InputsGrads[1]);
        else
            gradWrtB.CopyTo(m_InputsGrads[1]);
    }

}
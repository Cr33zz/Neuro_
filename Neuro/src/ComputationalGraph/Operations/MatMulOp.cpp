#include <algorithm>
#include "ComputationalGraph/Operations/MatMulOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    MatMulOp::MatMulOp(NodeBase* a, NodeBase* b)
        : Operation({a, b})
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void MatMulOp::ComputeInternal()
    {
        m_Output.Resize(Shape(m_Inputs[1]->Len(0), m_Inputs[0]->Len(1), m_Inputs[0]->Len(2), max(m_Inputs[0]->Len(3), m_Inputs[1]->Len(3))));
        m_Inputs[0]->Mul(*m_Inputs[1], m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void MatMulOp::ComputeGradientInternal(const Tensor& grad)
    {
        auto& a = *m_Inputs[0];
        auto& b = *m_Inputs[1];

        auto gradWrtA = grad.Mul(b.Transposed());

        if (m_InputsGrads[0].Len(3) != gradWrtA.Len(3))
            gradWrtA.Sum(BatchAxis, m_InputsGrads[0]);
        else
            gradWrtA.CopyTo(m_InputsGrads[0]);

        auto gradWrtB = a.Transposed().Mul(grad);

        if (m_InputsGrads[1].Len(3) != gradWrtB.Len(3))
            gradWrtB.Sum(BatchAxis, m_InputsGrads[1]);
        else
            gradWrtB.CopyTo(m_InputsGrads[1]);
    }

}
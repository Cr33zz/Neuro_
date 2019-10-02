#include <algorithm>
#include "ComputationalGraph/Operations/MultiplyOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    MultiplyOp::MultiplyOp(TensorLike* a, TensorLike* b)
        : Operation({a, b})
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void MultiplyOp::ComputeInternal()
    {
        m_Output.Resize(Shape(
            max(m_Inputs[0]->Len(0), m_Inputs[1]->Len(0)),
            max(m_Inputs[0]->Len(1), m_Inputs[1]->Len(1)),
            max(m_Inputs[0]->Len(2), m_Inputs[1]->Len(2)),
            max(m_Inputs[0]->Len(3), m_Inputs[1]->Len(3))));

        return m_Inputs[0]->MulElem(*m_Inputs[1], m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void MultiplyOp::ComputeGradientInternal(const Tensor& grad)
    {
        auto& a = *m_Inputs[0];
        auto& b = *m_Inputs[1];

        auto gradWrtA = grad * b;
        auto gradWrtB = grad * a;

        for (int i = WidthAxis; i <= BatchAxis; ++i)
        {
            if (gradWrtA.Len(i) != 1 && a.Len(i) == 1)
                gradWrtA = sum(gradWrtA, (EAxis)i);

            if (gradWrtB.Len(i) != 1 && b.Len(i) == 1)
                gradWrtB = sum(gradWrtB, (EAxis)i);
        }

        gradWrtA.CopyTo(m_InputsGrads[0]);
        gradWrtB.CopyTo(m_InputsGrads[1]);
    }
}
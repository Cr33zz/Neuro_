#include "ComputationalGraph/Operations/Subtract.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Op::Subtract::Subtract(NodeBase* a, NodeBase* b)
        : Operation({ a, b })
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Subtract::ComputeInternal()
    {
        return m_Inputs[0]->Sub(*m_Inputs[1], m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Subtract::ComputeGradientInternal(const Tensor& grad)
    {
        auto& a = *m_Inputs[0];
        auto& b = *m_Inputs[1];

        auto gradWrtA = grad;
        auto gradWrtB = grad.Negated();

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
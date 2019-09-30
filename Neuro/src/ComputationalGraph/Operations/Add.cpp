#include "ComputationalGraph/Operations/Add.h"

namespace Neuro
{        
    //////////////////////////////////////////////////////////////////////////
    Op::Add::Add(NodeBase* a, NodeBase* b)
        : Operation({ a, b })
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Add::ComputeInternal()
    {
        return m_Inputs[0]->Add(*m_Inputs[1], m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Add::ComputeGradientInternal(const Tensor& grad)
    {
        auto& a = *m_Inputs[0];
        auto& b = *m_Inputs[1];

        auto gradWrtA = grad;
        auto gradWrtB = grad;

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

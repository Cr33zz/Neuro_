#include "CompGraph/Operations/Add.h"

namespace Neuro
{    
    //////////////////////////////////////////////////////////////////////////
    Add::Add(NodeBase* a, NodeBase* b)
        : Operation({ a, b })
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Add::ComputeInternal()
    {
        return m_Inputs[0]->Add(*m_Inputs[1], m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Add::ComputeGradientInternal(const Tensor& grad)
    {
        auto a = m_Inputs[0];
        auto b = m_Inputs[1];

        auto& gradWrtA = m_InputsGrads[0];
        auto& gradWrtB = m_InputsGrads[1];

        gradWrtA = grad;
        gradWrtB = grad;

        for (int i = WidthAxis; i <= BatchAxis; ++i)
        {
            if (gradWrtA.Len(i) != 1 && a->Len(i) == 1)
                gradWrtA = sum(gradWrtA, (EAxis)i);

            if (gradWrtB.Len(i) != 1 && b->Len(i) == 1)
                gradWrtB = sum(gradWrtB, (EAxis)i);
        }
    }
}

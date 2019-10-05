#include <algorithm>
#include "ComputationalGraph/Operations/DivideOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    DivideOp::DivideOp(TensorLike* a, TensorLike* b, const string& name)
        : Operation({ a, b }, name.empty() ? "divide" : name)
    {
        m_Output.Resize(Shape(max(a->GetShape().Width(), b->GetShape().Width()), max(a->GetShape().Height(), b->GetShape().Height()), max(a->GetShape().Depth(), b->GetShape().Depth())));
    }

    //////////////////////////////////////////////////////////////////////////
    void DivideOp::ComputeInternal()
    {
        m_Output.ResizeBatch(max(m_Inputs[0]->Batch(), m_Inputs[1]->Batch()));
        return m_Inputs[0]->MulElem(*m_Inputs[1], m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void DivideOp::ComputeGradientInternal(const Tensor& grad)
    {
        auto& a = *m_Inputs[0];
        auto& b = *m_Inputs[1];

        auto gradWrtA = grad / b;
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
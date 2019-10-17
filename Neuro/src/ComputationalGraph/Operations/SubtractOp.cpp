#include <algorithm>
#include "ComputationalGraph/Operations/SubtractOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    SubtractOp::SubtractOp(TensorLike* a, TensorLike* b, const string& name)
        : Operation({ a, b }, name.empty() ? "sub" : name)
    {
        const Shape& aShape = a->GetShape();
        const Shape& bShape = b->GetShape();
        NEURO_ASSERT(aShape.Width() == bShape.Width() || aShape.Width() == 1 || bShape.Width() == 1, "Mismatched width " << aShape.Width() << " and " << bShape.Width());
        NEURO_ASSERT(aShape.Height() == bShape.Height() || aShape.Height() == 1 || bShape.Height() == 1, "Mismatched height " << aShape.Height() << " and " << bShape.Height());
        NEURO_ASSERT(aShape.Depth() == bShape.Depth() || aShape.Depth() == 1 || bShape.Depth() == 1, "Mismatched depth " << aShape.Depth() << " and " << bShape.Depth());
        NEURO_ASSERT(aShape.Batch() == bShape.Batch() || aShape.Batch() == 1 || bShape.Batch() == 1, "Mismatched batch " << aShape.Batch() << " and " << bShape.Batch());
        m_Output.Resize(Shape(max(aShape.Width(), bShape.Width()), max(aShape.Height(), bShape.Height()), max(aShape.Depth(), bShape.Depth())));
    }

    //////////////////////////////////////////////////////////////////////////
    void SubtractOp::ComputeInternal()
    {
        m_Output.ResizeBatch(max(m_Inputs[0]->Batch(), m_Inputs[1]->Batch()));
        return m_Inputs[0]->Sub(*m_Inputs[1], m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void SubtractOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
        {
            auto& a = *m_Inputs[0];
            auto gradWrtA = grad;
            for (int i = WidthAxis; i <= BatchAxis; ++i)
            {
                if (gradWrtA.Len(i) != 1 && a.Len(i) == 1)
                    gradWrtA = sum(gradWrtA, (EAxis)i);
            }
            gradWrtA.CopyTo(m_InputsGrads[0]);
        }

        if (m_InputNodes[1]->CareAboutGradient())
        {
            auto& b = *m_Inputs[1];
            auto gradWrtB = grad.Negated();
            for (int i = WidthAxis; i <= BatchAxis; ++i)
            {
                if (gradWrtB.Len(i) != 1 && b.Len(i) == 1)
                    gradWrtB = sum(gradWrtB, (EAxis)i);
            }
            gradWrtB.CopyTo(m_InputsGrads[1]);
        }
    }
}
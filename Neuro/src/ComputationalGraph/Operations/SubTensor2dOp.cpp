#include "ComputationalGraph/Operations/SubTensor2dOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    SubTensor2dOp::SubTensor2dOp(TensorLike* x, uint32_t width, uint32_t height, uint32_t widthOffset, uint32_t heightOffset, const string& name)
        : Operation({ x }, name.empty() ? "sub_tensor2d" : name), m_Width(width), m_Height(height), m_WidthOffset(widthOffset), m_HeightOffset(heightOffset)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void SubTensor2dOp::UpdateOutputShape()
    {
        NEURO_ASSERT(m_Width > 0, "");
        NEURO_ASSERT(m_Height > 0, "");
        const auto& shape = m_InputNodes[0]->GetShape();
        m_Output.Resize(Shape(m_Width, m_Height, shape.Depth(), shape.Batch()));
    }

    //////////////////////////////////////////////////////////////////////////
    void SubTensor2dOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->ExtractSubTensor2D(m_WidthOffset, m_HeightOffset, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void SubTensor2dOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
        {
            m_InputsGrads[0].Zero();
            grad.FuseSubTensor2D(m_WidthOffset, m_HeightOffset, m_InputsGrads[0]);
        }
    }
}
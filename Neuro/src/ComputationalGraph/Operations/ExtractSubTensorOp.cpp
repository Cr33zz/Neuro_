#include "ComputationalGraph/Operations/ExtractSubTensorOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    ExtractSubTensorOp::ExtractSubTensorOp(TensorLike* x, uint32_t width, uint32_t height, uint32_t widthOffset, uint32_t heightOffset, bool clampAllowed, const string& name)
        : Operation({ x }, name.empty() ? "extract_subtensor" : name), m_Width(width), m_Height(height), m_WidthOffset(widthOffset), m_HeightOffset(heightOffset), m_ClampAllowed(clampAllowed)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void ExtractSubTensorOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->ExtractSubTensor2D(m_WidthOffset, m_HeightOffset, m_Output, m_ClampAllowed);
    }

    //////////////////////////////////////////////////////////////////////////
    void ExtractSubTensorOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            grad.FuseSubTensor2D(m_WidthOffset, m_HeightOffset, m_InputsGrads[0], m_ClampAllowed);
    }

    //////////////////////////////////////////////////////////////////////////
    void ExtractSubTensorOp::UpdateOutputShape()
    {
        m_Output.Resize(Shape(m_Width, m_Height, m_Inputs[0]->Depth()));
    }
}
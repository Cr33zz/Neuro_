#include "ComputationalGraph/Operations/FuseSubTensorsOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    FuseSubTensorsOp::FuseSubTensorsOp(const vector<TensorLike*>& xs, size_t tX, size_t tY, const string& name)
        : Operation(xs, name.empty() ? "fuse_subtensors" : name), m_TX(tX), m_TY(tY)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void FuseSubTensorsOp::UpdateOutputShape()
    {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t depth = m_Inputs[0]->GetShape().Depth();

        size_t i = 0;
        for (uint32_t tY = 0; tY < m_TY; ++tY)
        {
            height += m_Inputs[i]->GetShape().Height();

            uint32_t tmpWidth = 0;
            for (uint32_t tX = 0; tX < m_TX; ++tX, ++i)
            {
                tmpWidth += m_Inputs[i]->GetShape().Width();
                NEURO_ASSERT(depth == m_Inputs[i]->GetShape().Depth(), "Depth mismatch between sub-tensors.");
            }

            if (width == 0)
                width = tmpWidth;
            else
                NEURO_ASSERT(width == tmpWidth, "Width mismatch between sub-tensors.");
        }

        m_Output.Resize(Shape(width, height, depth));
    }

    //////////////////////////////////////////////////////////////////////////
    void FuseSubTensorsOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());

        uint32_t widthOffset = 0;
        uint32_t heightOffset = 0;

        size_t i = 0;
        for (uint32_t tY = 0; tY < m_TY; ++tY)
        {
            widthOffset = 0;

            for (uint32_t tX = 0; tX < m_TX; ++tX, ++i)
            {
                m_Inputs[i]->FuseSubTensor2D(widthOffset, heightOffset, m_Output);
                widthOffset += m_Inputs[i]->GetShape().Width();
            }

            heightOffset += m_Inputs[0]->GetShape().Height();
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void FuseSubTensorsOp::ComputeGradientInternal(const Tensor& grad)
    {
        uint32_t widthOffset = 0;
        uint32_t heightOffset = 0;

        size_t i = 0;
        for (uint32_t tY = 0; tY < m_TY; ++tY)
        {
            widthOffset = 0;

            for (uint32_t tX = 0; tX < m_TX; ++tX, ++i)
            {
                if (m_InputNodes[i]->CareAboutGradient())
                    m_Output.ExtractSubTensor2D(widthOffset, heightOffset, m_InputsGrads[i]);
                widthOffset += m_Inputs[i]->GetShape().Width();
            }

            heightOffset += m_Inputs[0]->GetShape().Height();
        }
    }
}
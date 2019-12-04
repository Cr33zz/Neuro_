#include "ComputationalGraph/Operations/SwapRedBlueChannelsOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    SwapRedBlueChannelsOp::SwapRedBlueChannelsOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "swap_red_blue_channels" : name)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void SwapRedBlueChannelsOp::UpdateOutputShape()
    {
        NEURO_ASSERT(m_InputNodes[0]->GetShape().Depth() == 3, "Input tensor must have 3 channels.");
        __super::UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void SwapRedBlueChannelsOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        const Tensor& x = *m_Inputs[0];
        for (uint32_t n = 0; n < x.Batch(); ++n)
        {
            x.CopyDepthTo(0, n, 2, n, m_Output);
            x.CopyDepthTo(1, n, 1, n, m_Output);
            x.CopyDepthTo(2, n, 0, n, m_Output);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void SwapRedBlueChannelsOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
        {
            for (uint32_t n = 0; n < grad.Batch(); ++n)
            {
                grad.CopyDepthTo(0, n, 2, n, m_InputsGrads[0]);
                grad.CopyDepthTo(1, n, 1, n, m_InputsGrads[0]);
                grad.CopyDepthTo(2, n, 0, n, m_InputsGrads[0]);
            }
        }
    }
}
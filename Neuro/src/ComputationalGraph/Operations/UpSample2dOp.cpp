#include "ComputationalGraph/Operations/UpSample2dOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    UpSample2dOp::UpSample2dOp(TensorLike* x, int scaleFactor, const string& name)
        : Operation({ x }, name.empty() ? "upsample2d" : name), m_ScaleFactor(scaleFactor)
    {
        m_Output.Resize(Shape(x->GetShape().Width() * m_ScaleFactor, x->GetShape().Height() * m_ScaleFactor, x->GetShape().Depth()));
    }

    //////////////////////////////////////////////////////////////////////////
    void UpSample2dOp::ComputeInternal()
    {
        auto& x = *m_Inputs[0];
        m_Output.ResizeBatch(x.Batch());
        x.UpSample2D(m_ScaleFactor, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void UpSample2dOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            grad.UpSample2DGradient(grad, m_ScaleFactor, m_InputsGrads[0]);
    }
}
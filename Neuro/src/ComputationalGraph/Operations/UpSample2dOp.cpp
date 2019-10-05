#include "ComputationalGraph/Operations/UpSample2dOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    UpSample2dOp::UpSample2dOp(TensorLike* x, int scaleFactor, const string& name)
        : Operation({ x }, name.empty() ? "upsample2d" : name), m_ScaleFactor(scaleFactor)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void UpSample2dOp::ComputeInternal()
    {
        auto& x = *m_Inputs[0];
        m_Output.Resize(Shape(x.Width() * m_ScaleFactor, x.Height() * m_ScaleFactor, x.Depth(), x.Batch()));
        x.UpSample2D(m_ScaleFactor, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void UpSample2dOp::ComputeGradientInternal(const Tensor& grad)
    {
        grad.UpSample2DGradient(grad, m_ScaleFactor, m_InputsGrads[0]);
    }
}
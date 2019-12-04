#include "ComputationalGraph/Operations/ReshapeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    ReshapeOp::ReshapeOp(TensorLike* x, const Shape& shape, const string& name)
        : Operation({ x }, name.empty() ? "reshape" : name), m_Shape(shape)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void ReshapeOp::UpdateOutputShape()
    {
        m_Output.Resize(m_Shape);
    }

    //////////////////////////////////////////////////////////////////////////
    void ReshapeOp::ComputeInternal()
    {
        m_Inputs[0]->CopyTo(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void ReshapeOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            grad.CopyTo(m_InputsGrads[0]);
    }
}
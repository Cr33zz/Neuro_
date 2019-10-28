#include "ComputationalGraph/Operations/ReshapeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    ReshapeOp::ReshapeOp(TensorLike* x, const Shape& shape, const string& name)
        : Operation({ x }, name.empty() ? "reshape" : name), m_Shape(shape)
    {
        m_Output.Resize(shape);
    }

    //////////////////////////////////////////////////////////////////////////
    void ReshapeOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->CopyTo(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void ReshapeOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            grad.CopyTo(m_InputsGrads[0]);
    }
}
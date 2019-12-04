#include "ComputationalGraph/Operations/BatchFlattenOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    BatchFlattenOp::BatchFlattenOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "batch_flatten" : name)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchFlattenOp::UpdateOutputShape()
    {
        const auto& shape = m_InputNodes[0]->GetShape();
        m_Output.Resize(Shape(shape.Width() * shape.Height() * shape.Depth(), 1, 1, shape.Batch()));
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchFlattenOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->CopyTo(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchFlattenOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            grad.CopyTo(m_InputsGrads[0]);
    }
}
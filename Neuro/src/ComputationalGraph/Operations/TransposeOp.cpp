#include "ComputationalGraph/Operations/TransposeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    TransposeOp::TransposeOp(TensorLike* x, const vector<EAxis>& axes, const string& name)
        : Operation({ x }, name.empty() ? "transpose" : name)
    {
        m_Permutation = Tensor::FillUpTranposeAxis(axes);

        auto& shape = x->GetShape();
        m_Output.Resize(Shape(shape.Dimensions[m_Permutation[0]], shape.Dimensions[m_Permutation[1]], shape.Dimensions[m_Permutation[2]], shape.Dimensions[m_Permutation[3]]));
    }

    //////////////////////////////////////////////////////////////////////////
    void TransposeOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->Transpose(m_Permutation, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TransposeOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            grad.Transpose(m_Permutation, m_InputsGrads[0]);
    }
}
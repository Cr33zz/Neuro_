#include "ComputationalGraph/Operations/TransposeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    TransposeOp::TransposeOp(TensorLike* x, const vector<EAxis>& axes, const string& name)
        : Operation({ x }, name.empty() ? "transpose" : name)
    {
        m_UndeterminedOutputShape = true;
        m_Permutation = Tensor::FillUpTranposeAxis(axes);

        m_InvPermutation.resize(m_Permutation.size());
        for (size_t i = 0; i < m_Permutation.size(); ++i)
            m_InvPermutation[m_Permutation[i]] = (EAxis)i;

        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void TransposeOp::UpdateOutputShape()
    {
        const auto& shape = m_InputNodes[0]->GetShape();
        m_Output.Resize(Shape(shape.Dimensions[m_Permutation[0]], shape.Dimensions[m_Permutation[1]], shape.Dimensions[m_Permutation[2]], shape.Dimensions[m_Permutation[3]]));
    }

    //////////////////////////////////////////////////////////////////////////
    void TransposeOp::ComputeInternal()
    {
        auto& shape = m_Inputs[0]->GetShape();
        m_Output.Resize(Shape(shape.Dimensions[m_Permutation[0]], shape.Dimensions[m_Permutation[1]], shape.Dimensions[m_Permutation[2]], shape.Dimensions[m_Permutation[3]]));
        m_Inputs[0]->Transpose(m_Permutation, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TransposeOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            grad.Transpose(m_InvPermutation, m_InputsGrads[0]);
    }
}
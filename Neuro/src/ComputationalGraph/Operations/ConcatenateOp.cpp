#include "ComputationalGraph/Operations/ConcatenateOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    ConcatenateOp::ConcatenateOp(const vector<TensorLike*>& xs, EAxis axis, const string& name)
        : Operation(xs, name.empty() ? "concatenate" : name), m_Axis(axis)
    {
        /*if (m_Axis == WidthAxis)
            m_Output.Resize(Shape(xs[0]->GetShape().Width() * (uint32_t)xs.size(), xs[0]->GetShape().Height(), xs[0]->GetShape().Depth(), xs[0]->GetShape().Batch()));
        else if (m_Axis == HeightAxis)
            m_Output.Resize(Shape(xs[0]->GetShape().Width(), xs[0]->GetShape().Height() * (uint32_t)xs.size(), xs[0]->GetShape().Depth(), xs[0]->GetShape().Batch()));
        else if (m_Axis == DepthAxis)
            m_Output.Resize(Shape(xs[0]->GetShape().Width(), xs[0]->GetShape().Height(), xs[0]->GetShape().Depth() * (uint32_t)xs.size(), xs[0]->GetShape().Batch()));
        else if (m_Axis == BatchAxis)
            m_Output.Resize(Shape(xs[0]->GetShape().Width(), xs[0]->GetShape().Height(), xs[0]->GetShape().Depth(), xs[0]->GetShape().Batch() * (uint32_t)xs.size()));
        else
            assert(false);*/
        assert(axis >= WidthAxis && axis <= BatchAxis);
    }

    //////////////////////////////////////////////////////////////////////////
    void ConcatenateOp::ComputeInternal()
    {
        if (m_Axis == WidthAxis)
            m_Output.Resize(Shape(m_Inputs[0]->GetShape().Width() * (uint32_t)m_Inputs.size(), m_Inputs[0]->GetShape().Height(), m_Inputs[0]->GetShape().Depth(), m_Inputs[0]->GetShape().Batch()));
        else if (m_Axis == HeightAxis)
            m_Output.Resize(Shape(m_Inputs[0]->GetShape().Width(), m_Inputs[0]->GetShape().Height() * (uint32_t)m_Inputs.size(), m_Inputs[0]->GetShape().Depth(), m_Inputs[0]->GetShape().Batch()));
        else if (m_Axis == DepthAxis)
            m_Output.Resize(Shape(m_Inputs[0]->GetShape().Width(), m_Inputs[0]->GetShape().Height(), m_Inputs[0]->GetShape().Depth() * (uint32_t)m_Inputs.size(), m_Inputs[0]->GetShape().Batch()));
        else if (m_Axis == BatchAxis)
            m_Output.Resize(Shape(m_Inputs[0]->GetShape().Width(), m_Inputs[0]->GetShape().Height(), m_Inputs[0]->GetShape().Depth(), m_Inputs[0]->GetShape().Batch() * (uint32_t)m_Inputs.size()));
        else
            assert(false);

        //m_Output.Resize(Shape::From(m_Output.GetShape(), m_Output.Batch() * (uint32_t)m_Inputs.size()));
        Tensor::Concat(m_Axis, m_Inputs, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void ConcatenateOp::ComputeGradientInternal(const Tensor& grad)
    {
        grad.Split(m_Axis, m_InputsGradsPtrs);
    }
}
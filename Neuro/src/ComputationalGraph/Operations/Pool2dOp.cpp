#include "ComputationalGraph/Operations/Pool2dOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Pool2dOp::Pool2dOp(TensorLike* x, uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, EDataFormat dataFormat, const string& name)
        : Operation({ x }, name.empty() ? "pool2d" : name), m_FilterSize(filterSize), m_Stride(stride), m_Padding(padding), m_DataFormat(dataFormat), m_Mode(mode)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void Pool2dOp::UpdateOutputShape()
    {
        auto x = m_InputNodes[0];
        const auto& shape = x->GetShape();
        m_Output.Resize(Shape::From(Tensor::GetPooling2DOutputShape(x->GetShape(), m_FilterSize, m_FilterSize, m_Stride, m_Padding, m_Padding, m_DataFormat), shape.Batch()));
    }

    //////////////////////////////////////////////////////////////////////////
    void Pool2dOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->Pool2D(m_FilterSize, m_Stride, m_Mode, m_Padding, m_DataFormat, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Pool2dOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            m_Inputs[0]->Pool2DGradient(m_Output, *m_Inputs[0], grad, m_FilterSize, m_Stride, m_Mode, m_Padding, m_DataFormat, m_InputsGrads[0]);
    }
}
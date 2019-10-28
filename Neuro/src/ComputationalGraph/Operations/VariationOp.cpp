#include "ComputationalGraph/Operations/VariationOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    VariationOp::VariationOp(TensorLike* x, EDataFormat dataFormat, const string& name)
        : Operation({ x }, name.empty() ? "variation" : name), m_DataFormat(dataFormat)
    {
        //single kernel to compute sum of horizontal and vertical pixels' values differences across 3 channels
        m_Kernel = Tensor({2.f, -1.f, -1.f, 0.f, 2.f, -1.f, -1.f, 0.f, 2.f, -1.f, -1.f, 0.f }, Shape(2,2,3), "variation_kernel");
        m_Output.Resize(Tensor::GetConvOutputShape(x->GetShape(), m_Kernel.Batch(), m_Kernel.Width(), m_Kernel.Height(), 1, 0, 0, m_DataFormat));
    }

    //////////////////////////////////////////////////////////////////////////
    void VariationOp::ComputeInternal()
    {
        auto& x = *m_Inputs[0];

        m_Output.ResizeBatch(x.Batch());

        return x.Conv2D(m_Kernel, 1, 0, m_DataFormat, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void VariationOp::ComputeGradientInternal(const Tensor& grad)
    {
        auto& x = *m_Inputs[0];
        
        if (m_InputNodes[0]->CareAboutGradient())
            grad.Conv2DInputsGradient(grad, m_Kernel, 1, 0, m_DataFormat, m_InputsGrads[0]);
    }
}

#include "ComputationalGraph/Operations/Conv2dOp.h"

namespace Neuro
{        
    //////////////////////////////////////////////////////////////////////////
    Conv2dOp::Conv2dOp(TensorLike* x, TensorLike* kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat, const string& name)
        : Operation({ x, kernels }, name.empty() ? "conv2d" : name), m_Stride(stride), m_Padding(padding), m_DataFormat(dataFormat)
    {
        m_Output.Resize(Tensor::GetConvOutputShape(x->GetShape(), kernels->GetShape().Batch(), kernels->GetShape().Width(), kernels->GetShape().Height(), m_Stride, m_Padding, m_Padding, m_DataFormat));
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2dOp::ComputeInternal()
    {
        auto& x = *m_Inputs[0];
        auto& kernels = *m_Inputs[1];

        m_Output.ResizeBatch(x.Batch());

        return x.Conv2D(kernels, m_Stride, m_Padding, m_DataFormat, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2dOp::ComputeGradientInternal(const Tensor& grad)
    {
        auto& x = *m_Inputs[0];
        auto& kernels = *m_Inputs[1];

        if (m_InputNodes[0]->CareAboutGradient())
            grad.Conv2DInputsGradient(grad, kernels, m_Stride, m_Padding, m_DataFormat, m_InputsGrads[0]);
        if (m_InputNodes[1]->CareAboutGradient())
            grad.Conv2DKernelsGradient(x, grad, m_Stride, m_Padding, m_DataFormat, m_InputsGrads[1]);
    }
}

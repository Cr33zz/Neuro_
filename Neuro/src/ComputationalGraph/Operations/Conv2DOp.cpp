#include "ComputationalGraph/Operations/Conv2DOp.h"

namespace Neuro
{        
    //////////////////////////////////////////////////////////////////////////
    Conv2DOp::Conv2DOp(NodeBase* x, NodeBase* kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat)
        : Operation({ x, kernels }), m_Stride(stride), m_Padding(padding), m_DataFormat(dataFormat)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DOp::ComputeInternal()
    {
        auto& x = *m_Inputs[0];
        auto& kernels = *m_Inputs[1];

        m_Output.Resize(Tensor::GetConvOutputShape(x.GetShape(), kernels.Batch(), kernels.Width(), kernels.Height(), m_Stride, m_Padding, m_Padding, m_DataFormat));

        return x.Conv2D(kernels, m_Stride, m_Padding, m_DataFormat, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DOp::ComputeGradientInternal(const Tensor& grad)
    {
        auto& x = *m_Inputs[0];
        auto& kernels = *m_Inputs[1];

        grad.Conv2DInputsGradient(grad, kernels, m_Stride, m_Padding, m_DataFormat, m_InputsGrads[0]);
        grad.Conv2DKernelsGradient(x, grad, m_Stride, m_Padding, m_DataFormat, m_InputsGrads[1]);
    }
}

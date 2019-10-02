#include "Layers/Conv2DTranspose.h"
#include "Layers/Conv2D.h"
#include "Tensors/Tensor.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose::Conv2DTranspose(LayerBase* inputLayer, uint32_t outputDepth, uint32_t filterSize, uint32_t stride, uint32_t padding, ActivationBase* activation, EDataFormat dataFormat, const string& name)
        : SingleLayer(__FUNCTION__, inputLayer, Tensor::GetConvTransposeOutputShape(inputLayer->OutputShape(), outputDepth, filterSize, filterSize, stride, padding, padding, dataFormat), activation, name)
    {
        m_FilterSize = filterSize;
        m_OutputDepth = outputDepth;
        m_Stride = stride;
        m_Padding = padding;
        m_DataFormat = dataFormat;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose::Conv2DTranspose(uint32_t outputDepth, uint32_t filterSize, uint32_t stride, uint32_t padding, ActivationBase* activation, EDataFormat dataFormat, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), activation, name)
    {
        m_FilterSize = filterSize;
        m_OutputDepth = outputDepth;
        m_Stride = stride;
        m_Padding = padding;
        m_DataFormat = dataFormat;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose::Conv2DTranspose(const Shape& inputShape, uint32_t outputDepth, uint32_t filterSize, uint32_t stride, uint32_t padding, ActivationBase* activation, EDataFormat dataFormat, const string& name)
        : SingleLayer(__FUNCTION__, inputShape, Tensor::GetConvTransposeOutputShape(inputShape, outputDepth, filterSize, filterSize, stride, padding, padding, dataFormat), activation, name)
    {
        m_FilterSize = filterSize;
        m_OutputDepth = outputDepth;
        m_Stride = stride;
        m_Padding = padding;
        m_DataFormat = dataFormat;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose::~Conv2DTranspose()
    {
        delete m_KernelInitializer;
        delete m_BiasInitializer;
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::InitOps(TensorLike* training, bool initValues)
    {
        if (m_DataFormat == NCHW)
        {
            m_Kernels = new Variable(Shape(m_FilterSize, m_FilterSize, m_OutputDepth, InputShape().Depth()), initValues ? m_KernelInitializer : nullptr, "kernels");
            m_Bias = new Variable(Shape(1, 1, m_OutputDepth), initValues ? m_BiasInitializer : nullptr, "bias");
        }
        else
        {
            m_Kernels = new Variable(Shape(m_FilterSize, m_FilterSize, m_OutputDepth, InputShape().Len(0)), initValues ? m_KernelInitializer : nullptr, "kernels");
            m_Bias = new Variable(Shape(m_OutputDepth), initValues ? m_BiasInitializer : nullptr, "bias");
        }

        m_OutputOps[0] = conv2d_transpose(m_InputOps[0], m_Kernels, m_Stride, m_Padding, m_DataFormat);
        if (m_UseBias)
            m_OutputOps[0] = add(m_OutputOps[0], m_Bias);
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::OnLinkInput(const vector<LayerBase*>& inputLayers)
    {
        __super::OnLinkInput(inputLayers);

        m_OutputsShapes[0] = Tensor::GetConvTransposeOutputShape(inputLayers[0]->OutputShape(), m_OutputDepth, m_FilterSize, m_FilterSize, m_Stride, m_Padding, m_Padding, m_DataFormat);
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Conv2DTranspose::GetCloneInstance() const
    {
        return new Conv2DTranspose();
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::OnClone(const LayerBase& source)
    {
        __super::OnClone(source);

        auto& sourceDeconv = static_cast<const Conv2DTranspose&>(source);
        m_Kernels = new Variable(*sourceDeconv.m_Kernels);
        m_Bias = new Variable(*sourceDeconv.m_Bias);
        m_UseBias = sourceDeconv.m_UseBias;
        m_FilterSize = sourceDeconv.m_FilterSize;
        m_OutputDepth = sourceDeconv.m_OutputDepth;
        m_Stride = sourceDeconv.m_Stride;
        m_Padding = sourceDeconv.m_Padding;
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::ParametersAndGradients(vector<ParameterAndGradient>& paramsAndGrads, bool onlyTrainable)
    {
        if (onlyTrainable && !m_Trainable)
            return;

        paramsAndGrads.push_back({ &m_Kernels->Output() });

        if (m_UseBias)
            paramsAndGrads.push_back({ &m_Bias->Output() });
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::SerializedParameters(vector<SerializedParameter>& params)
    {
        params.push_back({ &m_Kernels->Output(), { DepthAxis, BatchAxis, HeightAxis, WidthAxis } });

        if (m_UseBias)
        {
            if (m_DataFormat == NCHW)
                params.push_back({ &m_Bias->Output(), { DepthAxis, HeightAxis, WidthAxis } });
            else
                params.push_back({ &m_Bias->Output() });
        }
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::Tensor& Conv2DTranspose::Kernels()
    {
        return m_Kernels->Output();
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::Tensor& Conv2DTranspose::Bias()
    {
        return m_Bias->Output();
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::CopyParametersTo(LayerBase& target, float tau) const
    {
        __super::CopyParametersTo(target, tau);

        auto& targetConv = static_cast<Conv2DTranspose&>(target);
        m_Kernels->Output().CopyTo(targetConv.m_Kernels->Output(), tau);
        m_Bias->Output().CopyTo(targetConv.m_Bias->Output(), tau);
    }

    //////////////////////////////////////////////////////////////////////////
    uint32_t Conv2DTranspose::ParamsNum() const
    {
        return m_FilterSize * m_FilterSize * m_OutputDepth + (m_UseBias ? m_OutputDepth : 0);
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose* Conv2DTranspose::KernelInitializer(InitializerBase* initializer)
    {
        delete m_KernelInitializer;
        m_KernelInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose* Conv2DTranspose::BiasInitializer(InitializerBase* initializer)
    {
        delete m_BiasInitializer;
        m_BiasInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose* Conv2DTranspose::UseBias(bool useBias)
    {
        m_UseBias = useBias;
        return this;
    }
}

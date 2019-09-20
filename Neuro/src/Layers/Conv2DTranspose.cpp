#include "Layers/Conv2DTranspose.h"
#include "Layers/Conv2D.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose::Conv2DTranspose(LayerBase* inputLayer, uint32_t filterSize, uint32_t outputDepth, uint32_t stride, uint32_t padding, ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, inputLayer, Tensor::GetConvTransposeOutputShape(inputLayer->OutputShape(), outputDepth, filterSize, filterSize, stride, padding, padding), activation, name)
    {
        m_FilterSize = filterSize;
        m_OutputDepth = outputDepth;
        m_Stride = stride;
        m_Padding = padding;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose::Conv2DTranspose(uint32_t filterSize, uint32_t outputDepth, uint32_t stride, uint32_t padding, ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), activation, name)
    {
        m_FilterSize = filterSize;
        m_OutputDepth = outputDepth;
        m_Stride = stride;
        m_Padding = padding;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose::Conv2DTranspose(const Shape& inputShape, uint32_t filterSize, uint32_t outputDepth, uint32_t stride, uint32_t padding, ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, inputShape, Tensor::GetConvTransposeOutputShape(inputShape, outputDepth, filterSize, filterSize, stride, padding, padding), activation, name)
    {
        m_FilterSize = filterSize;
        m_OutputDepth = outputDepth;
        m_Stride = stride;
        m_Padding = padding;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose::~Conv2DTranspose()
    {
        delete m_KernelInitializer;
        delete m_BiasInitializer;
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::OnInit()
    {
        __super::OnInit();

        m_Kernels = Tensor(Shape(m_FilterSize, m_FilterSize, m_OutputDepth, InputShape().Depth()), Name() + "/kernels");
        m_Bias = Tensor(Shape(1, 1, m_OutputDepth), Name() + "/bias");
        m_KernelsGradient = Tensor(m_Kernels.GetShape(), Name() + "/kernels_grad");
        m_KernelsGradient.Zero();
        m_BiasGradient = Tensor(m_Bias.GetShape(), Name() + "/bias_grad");
        m_BiasGradient.Zero();

        m_KernelInitializer->Init(m_Kernels);
        if (m_UseBias)
            m_BiasInitializer->Init(m_Bias);
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::OnLink(LayerBase* layer, bool input)
    {
        __super::OnLink(layer, input);

        if (input)
            m_OutputShapes[0] = Tensor::GetConvTransposeOutputShape(layer->OutputShape(), m_OutputDepth, m_FilterSize, m_FilterSize, m_Stride, m_Padding, m_Padding);
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
        m_Kernels = Tensor(sourceDeconv.m_Kernels);
        m_Bias = Tensor(sourceDeconv.m_Bias);
        m_UseBias = sourceDeconv.m_UseBias;
        m_FilterSize = sourceDeconv.m_FilterSize;
        m_OutputDepth = sourceDeconv.m_OutputDepth;
        m_Stride = sourceDeconv.m_Stride;
        m_Padding = sourceDeconv.m_Padding;
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::FeedForwardInternal(bool training)
    {
        m_Inputs[0]->Conv2DTransposed(m_Kernels, m_Stride, m_Padding, m_Outputs[0]);
        if (m_UseBias)
            m_Outputs[0].Add(m_Bias, m_Outputs[0]);
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::BackPropInternal(vector<Tensor>& outputsGradient)
    {
        outputsGradient[0].Conv2DTransposedInputsGradient(outputsGradient[0], m_Kernels, m_Stride, m_Padding, m_InputsGradient[0]);

        if (m_Trainable)
        {
            outputsGradient[0].Conv2DTransposedKernelsGradient(*m_Inputs[0], outputsGradient[0], m_Stride, m_Padding, m_KernelsGradient);
            if (m_UseBias)
                m_BiasGradient.Add(outputsGradient[0].Sum(EAxis::WHBAxis), m_BiasGradient);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::GetParametersAndGradients(vector<ParametersAndGradients>& paramsAndGrads, bool onlyTrainable)
    {
        if (onlyTrainable && !m_Trainable)
            return;

        paramsAndGrads.push_back(ParametersAndGradients(&m_Kernels, &m_KernelsGradient));

        if (m_UseBias)
            paramsAndGrads.push_back(ParametersAndGradients(&m_Bias, &m_BiasGradient));
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::CopyParametersTo(LayerBase& target, float tau) const
    {
        __super::CopyParametersTo(target, tau);

        auto& targetConv = static_cast<Conv2DTranspose&>(target);
        m_Kernels.CopyTo(targetConv.m_Kernels, tau);
        m_Bias.CopyTo(targetConv.m_Bias, tau);
    }

    //////////////////////////////////////////////////////////////////////////
    uint32_t Conv2DTranspose::ParamsNum() const
    {
        return m_FilterSize * m_FilterSize * m_OutputDepth + (m_UseBias ? m_OutputDepth : 0);
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose* Conv2DTranspose::SetKernelInitializer(InitializerBase* initializer)
    {
        delete m_KernelInitializer;
        m_KernelInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose* Conv2DTranspose::SetBiasInitializer(InitializerBase* initializer)
    {
        delete m_BiasInitializer;
        m_BiasInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose* Conv2DTranspose::SetUseBias(bool useBias)
    {
        m_UseBias = useBias;
        return this;
    }
}

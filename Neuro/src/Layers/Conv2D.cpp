#include "Layers/Conv2D.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
    Conv2D::Conv2D(LayerBase* inputLayer, uint32_t filtersNum, uint32_t filterSize, uint32_t stride, uint32_t padding, ActivationBase* activation, EDataFormat dataFormat, const string& name)
		: SingleLayer(__FUNCTION__, inputLayer, Tensor::GetConvOutputShape(inputLayer->OutputShape(), filtersNum, filterSize, filterSize, stride, padding, padding, dataFormat), activation, name)
	{
		m_FilterSize = filterSize;
		m_FiltersNum = filtersNum;
		m_Stride = stride;
        m_Padding = padding;
        m_DataFormat = dataFormat;
    }

	//////////////////////////////////////////////////////////////////////////
    Conv2D::Conv2D(uint32_t filtersNum, uint32_t filterSize, uint32_t stride, uint32_t padding, ActivationBase* activation, EDataFormat dataFormat, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), activation, name)
    {
        m_FilterSize = filterSize;
        m_FiltersNum = filtersNum;
        m_Stride = stride;
        m_Padding = padding;
        m_DataFormat = dataFormat;
    }

	//////////////////////////////////////////////////////////////////////////
	Conv2D::Conv2D(const Shape& inputShape, uint32_t filtersNum, uint32_t filterSize, uint32_t stride, uint32_t padding, ActivationBase* activation, EDataFormat dataFormat, const string& name)
		: SingleLayer(__FUNCTION__, inputShape, Tensor::GetConvOutputShape(inputShape, filtersNum, filterSize, filterSize, stride, padding, padding, dataFormat), activation, name)
	{
		m_FilterSize = filterSize;
		m_FiltersNum = filtersNum;
		m_Stride = stride;
        m_Padding = padding;
        m_DataFormat = dataFormat;
	}

    //////////////////////////////////////////////////////////////////////////
	Conv2D::~Conv2D()
	{
		delete m_KernelInitializer;
		delete m_BiasInitializer;
	}

	//////////////////////////////////////////////////////////////////////////
	void Conv2D::OnInit(bool initValues)
	{
		__super::OnInit(initValues);

        if (m_DataFormat == NCHW)
        {
            m_Kernels = Tensor(Shape(m_FilterSize, m_FilterSize, InputShape().Depth(), m_FiltersNum), Name() + "/kernels");
            m_Bias = Tensor(Shape(1, 1, m_FiltersNum), Name() + "/bias");
        }
        else
        {
            m_Kernels = Tensor(Shape(m_FilterSize, m_FilterSize, InputShape().Len(0), m_FiltersNum), Name() + "/kernels");
            m_Bias = Tensor(Shape(m_FiltersNum), Name() + "/bias");
        }
		m_KernelsGradient = Tensor(m_Kernels.GetShape(), Name() + "/kernels_grad");
        m_KernelsGradient.Zero();
		m_BiasGradient = Tensor(m_Bias.GetShape(), Name() + "/bias_grad");
        m_BiasGradient.Zero();

        if (initValues)
        {
            m_KernelInitializer->Init(m_Kernels);
            if (m_UseBias)
                m_BiasInitializer->Init(m_Bias);
        }
	}

    //////////////////////////////////////////////////////////////////////////
    void Conv2D::OnLinkInput(const vector<LayerBase*>& inputLayers)
    {
        __super::OnLinkInput(inputLayers);

        m_OutputsShapes[0] = Tensor::GetConvOutputShape(inputLayers[0]->OutputShape(), m_FiltersNum, m_FilterSize, m_FilterSize, m_Stride, m_Padding, m_Padding, m_DataFormat);
    }

    //////////////////////////////////////////////////////////////////////////
	LayerBase* Conv2D::GetCloneInstance() const
	{
		return new Conv2D();
	}

	//////////////////////////////////////////////////////////////////////////
	void Conv2D::OnClone(const LayerBase& source)
	{
		__super::OnClone(source);

		auto& sourceConv = static_cast<const Conv2D&>(source);
		m_Kernels = Tensor(sourceConv.m_Kernels);
		m_Bias = Tensor(sourceConv.m_Bias);
		m_UseBias = sourceConv.m_UseBias;
		m_FilterSize = sourceConv.m_FilterSize;
		m_FiltersNum = sourceConv.m_FiltersNum;
		m_Stride = sourceConv.m_Stride;
	}

	//////////////////////////////////////////////////////////////////////////
	void Conv2D::FeedForwardInternal(bool training)
	{
		m_Inputs[0]->Conv2D(m_Kernels, m_Stride, m_Padding, m_DataFormat, *m_Outputs[0]);
		if (m_UseBias)
			m_Outputs[0]->Add(m_Bias, *m_Outputs[0]);
	}

	//////////////////////////////////////////////////////////////////////////
	void Conv2D::BackPropInternal(const tensor_ptr_vec_t& outputsGradient)
	{
		outputsGradient[0]->Conv2DInputsGradient(*outputsGradient[0], m_Kernels, m_Stride, m_Padding, m_DataFormat, *m_InputsGradient[0]);

        if (m_Trainable)
        {
            outputsGradient[0]->Conv2DKernelsGradient(*m_Inputs[0], *outputsGradient[0], m_Stride, m_Padding, m_DataFormat, m_KernelsGradient);
            if (m_UseBias)
                m_BiasGradient.Add(outputsGradient[0]->Sum(m_DataFormat == NCHW ? _013Axes : _123Axes), m_BiasGradient);
        }
	}

    //////////////////////////////////////////////////////////////////////////
	void Conv2D::ParametersAndGradients(vector<ParameterAndGradient>& paramsAndGrads, bool onlyTrainable)
	{
        if (onlyTrainable && !m_Trainable)
            return;

        paramsAndGrads.push_back({ &m_Kernels, &m_KernelsGradient });

		if (m_UseBias)
            paramsAndGrads.push_back({ &m_Bias, &m_BiasGradient });
	}

    //////////////////////////////////////////////////////////////////////////
    void Conv2D::SerializedParameters(vector<SerializedParameter>& params)
    {
        params.push_back({ &m_Kernels, { DepthAxis, BatchAxis, HeightAxis, WidthAxis } });

        if (m_UseBias)
        {
            if (m_DataFormat == NCHW)
                params.push_back({ &m_Bias, { DepthAxis, HeightAxis, WidthAxis } });
            else
                params.push_back({ &m_Bias });
        }
    }

	//////////////////////////////////////////////////////////////////////////
	void Conv2D::CopyParametersTo(LayerBase& target, float tau) const
	{
		__super::CopyParametersTo(target, tau);

		auto& targetConv = static_cast<Conv2D&>(target);
		m_Kernels.CopyTo(targetConv.m_Kernels, tau);
		m_Bias.CopyTo(targetConv.m_Bias, tau);
	}

	//////////////////////////////////////////////////////////////////////////
	uint32_t Conv2D::ParamsNum() const
	{
		return m_FilterSize * m_FilterSize * (m_DataFormat == NCHW ? InputShape().Depth() : InputShape().Len(0)) * m_FiltersNum + (m_UseBias ? m_FiltersNum : 0);
	}

    //////////////////////////////////////////////////////////////////////////
    Conv2D* Conv2D::KernelInitializer(InitializerBase* initializer)
    {
        delete m_KernelInitializer;
        m_KernelInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2D* Conv2D::BiasInitializer(InitializerBase* initializer)
    {
        delete m_BiasInitializer;
        m_BiasInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2D* Conv2D::UseBias(bool useBias)
    {
        m_UseBias = useBias;
        return this;
    }
}

#include "Layers/Conv2D.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Conv2D::Conv2D(LayerBase* inputLayer, uint filterSize, uint filtersNum, uint stride, uint padding, ActivationBase* activation, const string& name)
		: LayerBase(__FUNCTION__, inputLayer, Tensor::GetConvOutputShape(inputLayer->OutputShape(), filtersNum, filterSize, filterSize, stride, padding, padding), activation, name)
	{
		m_FilterSize = filterSize;
		m_FiltersNum = filtersNum;
		m_Stride = stride;
        m_Padding = padding;
    }

	//////////////////////////////////////////////////////////////////////////
	Conv2D::Conv2D(const Shape& inputShape, uint filterSize, uint filtersNum, uint stride, uint padding, ActivationBase* activation, const string& name)
		: LayerBase(__FUNCTION__, inputShape, Tensor::GetConvOutputShape(inputShape, filtersNum, filterSize, filterSize, stride, padding, padding), activation, name)
	{
		m_FilterSize = filterSize;
		m_FiltersNum = filtersNum;
		m_Stride = stride;
        m_Padding = padding;
	}

	//////////////////////////////////////////////////////////////////////////
	Conv2D::Conv2D()
	{
	}

	//////////////////////////////////////////////////////////////////////////
	Conv2D::~Conv2D()
	{
		delete m_KernelInitializer;
		delete m_BiasInitializer;
	}

	//////////////////////////////////////////////////////////////////////////
	void Conv2D::OnInit()
	{
		__super::OnInit();

		m_Kernels = Tensor(Shape(m_FilterSize, m_FilterSize, InputShape().Depth(), m_FiltersNum), Name() + "/kernels");
		m_Bias = Tensor(Shape(m_OutputShape.Width(), m_OutputShape.Height(), m_FiltersNum), Name() + "/bias");
		m_KernelsGradient = Tensor(m_Kernels.GetShape(), Name() + "/kernels_grad");
		m_BiasGradient = Tensor(m_Bias.GetShape(), Name() + "/bias_grad");

		m_KernelInitializer->Init(m_Kernels, m_InputShapes[0].Length, m_OutputShape.Length);
		if (m_UseBias)
			m_BiasInitializer->Init(m_Bias, m_InputShapes[0].Length, m_OutputShape.Length);
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
		m_Inputs[0]->Conv2D(m_Kernels, m_Stride, m_Padding, m_Output);
		if (m_UseBias)
			m_Output.Add(m_Bias, m_Output);
	}

	//////////////////////////////////////////////////////////////////////////
	void Conv2D::BackPropInternal(Tensor& outputGradient)
	{
		outputGradient.Conv2DInputsGradient(outputGradient, m_Kernels, m_Stride, m_Padding, m_InputsGradient[0]);
		outputGradient.Conv2DKernelsGradient(*m_Inputs[0], outputGradient, m_Stride, m_Padding, m_KernelsGradient);

		if (m_UseBias)
			m_BiasGradient.Add(outputGradient.Sum(EAxis::Feature));
	}

    /*Neuro::EPaddingMode Convolution::GetGradientPaddingMode(uint padding)
    {
        if (paddingMode == EPaddingMode::Valid)
            return EPaddingMode::Full;
        if (paddingMode == EPaddingMode::Full)
            return EPaddingMode::Valid;
        return paddingMode;
    }*/

    //////////////////////////////////////////////////////////////////////////
	void Conv2D::GetParametersAndGradients(vector<ParametersAndGradients>& result)
	{
		result.push_back(ParametersAndGradients(&m_Kernels, &m_KernelsGradient));

		if (m_UseBias)
			result.push_back(ParametersAndGradients(&m_Bias, &m_BiasGradient));
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
	int Conv2D::GetParamsNum() const
	{
		return m_FilterSize * m_FilterSize * m_FiltersNum;
	}

    //////////////////////////////////////////////////////////////////////////
    Conv2D* Conv2D::SetKernelInitializer(InitializerBase* initializer)
    {
        delete m_KernelInitializer;
        m_KernelInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2D* Conv2D::SetBiasInitializer(InitializerBase* initializer)
    {
        delete m_BiasInitializer;
        m_BiasInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2D* Conv2D::SetUseBias(bool useBias)
    {
        m_UseBias = useBias;
        return this;
    }
}

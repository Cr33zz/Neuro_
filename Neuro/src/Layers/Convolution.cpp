#include "Layers/Convolution.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Convolution::Convolution(LayerBase* inputLayer, int filterSize, int filtersNum, int stride, Tensor::EPaddingType paddingMode, ActivationBase* activation, const string& name)
		: LayerBase(__FUNCTION__, inputLayer, GetOutShape(inputLayer->OutputShape(), filterSize, filterSize, stride, filtersNum), activation, name)
	{
		m_FilterSize = filterSize;
		m_FiltersNum = filtersNum;
		m_Stride = stride;
        m_PaddingMode = paddingMode;
    }

	//////////////////////////////////////////////////////////////////////////
	Convolution::Convolution(const Shape& inputShape, int filterSize, int filtersNum, int stride, Tensor::EPaddingType paddingMode, ActivationBase* activation, const string& name)
		: LayerBase(__FUNCTION__, inputShape, GetOutShape(inputShape, filterSize, filterSize, stride, filtersNum), activation, name)
	{
		m_FilterSize = filterSize;
		m_FiltersNum = filtersNum;
		m_Stride = stride;
        m_PaddingMode = paddingMode;
	}

	//////////////////////////////////////////////////////////////////////////
	Convolution::Convolution()
	{
	}

	//////////////////////////////////////////////////////////////////////////
	Convolution::~Convolution()
	{
		delete m_KernelInitializer;
		delete m_BiasInitializer;
	}

	//////////////////////////////////////////////////////////////////////////
	void Convolution::OnInit()
	{
		__super::OnInit();

		m_Kernels = Tensor(Shape(m_FilterSize, m_FilterSize, InputShape().Depth(), m_FiltersNum));
		m_Bias = Tensor(Shape(m_OutputShape.Width(), m_OutputShape.Height(), m_FiltersNum));
		m_KernelsGradient = Tensor(m_Kernels.GetShape());
		m_BiasGradient = Tensor(m_Bias.GetShape());

		m_KernelInitializer->Init(m_Kernels, m_InputShapes[0].Length, m_OutputShape.Length);
		if (m_UseBias)
			m_BiasInitializer->Init(m_Bias, m_InputShapes[0].Length, m_OutputShape.Length);
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase* Convolution::GetCloneInstance() const
	{
		return new Convolution();
	}

	//////////////////////////////////////////////////////////////////////////
	void Convolution::OnClone(const LayerBase& source)
	{
		__super::OnClone(source);

		auto& sourceConv = static_cast<const Convolution&>(source);
		m_Kernels = Tensor(sourceConv.m_Kernels);
		m_Bias = Tensor(sourceConv.m_Bias);
		m_UseBias = sourceConv.m_UseBias;
		m_FilterSize = sourceConv.m_FilterSize;
		m_FiltersNum = sourceConv.m_FiltersNum;
		m_Stride = sourceConv.m_Stride;
	}

	//////////////////////////////////////////////////////////////////////////
	void Convolution::FeedForwardInternal(bool training)
	{
		m_Inputs[0]->Conv2D(m_Kernels, m_Stride, m_PaddingMode, m_Output);
		if (m_UseBias)
			m_Output.Add(m_Bias, m_Output);
	}

	//////////////////////////////////////////////////////////////////////////
	void Convolution::BackPropInternal(Tensor& outputGradient)
	{
		outputGradient.Conv2DInputsGradient(outputGradient, m_Kernels, m_Stride, GetGradientPaddingMode(m_PaddingMode), m_InputsGradient[0]);
		outputGradient.Conv2DKernelsGradient(*m_Inputs[0], outputGradient, m_Stride, m_PaddingMode, m_KernelsGradient);

		if (m_UseBias)
			m_BiasGradient.Add(outputGradient.SumBatches());
	}

	//////////////////////////////////////////////////////////////////////////
	Shape Convolution::GetOutShape(const Shape& inputShape, int filterWidth, int filterHeight, int stride, int filtersNum)
	{
		return Shape((int)floor((float)(inputShape.Width() - filterWidth) / stride + 1), (int)floor((float)(inputShape.Height() - filterHeight) / stride + 1), filtersNum);
	}

    Neuro::Tensor::EPaddingType Convolution::GetGradientPaddingMode(Tensor::EPaddingType paddingMode)
    {
        if (paddingMode == Tensor::EPaddingType::Valid)
            return Tensor::EPaddingType::Full;
        if (paddingMode == Tensor::EPaddingType::Full)
            return Tensor::EPaddingType::Valid;
        return paddingMode;
    }

    //////////////////////////////////////////////////////////////////////////
	void Convolution::GetParametersAndGradients(vector<ParametersAndGradients>& result)
	{
		result.push_back(ParametersAndGradients(&m_Kernels, &m_KernelsGradient));

		if (m_UseBias)
			result.push_back(ParametersAndGradients(&m_Bias, &m_BiasGradient));
	}

	//////////////////////////////////////////////////////////////////////////
	void Convolution::CopyParametersTo(LayerBase& target, float tau) const
	{
		__super::CopyParametersTo(target, tau);

		auto& targetConv = static_cast<Convolution&>(target);
		m_Kernels.CopyTo(targetConv.m_Kernels, tau);
		m_Bias.CopyTo(targetConv.m_Bias, tau);
	}

	//////////////////////////////////////////////////////////////////////////
	int Convolution::GetParamsNum() const
	{
		return m_FilterSize * m_FilterSize * m_FiltersNum;
	}

    //////////////////////////////////////////////////////////////////////////
    Convolution* Convolution::SetKernelInitializer(InitializerBase* initializer)
    {
        delete m_KernelInitializer;
        m_KernelInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Convolution* Convolution::SetBiasInitializer(InitializerBase* initializer)
    {
        delete m_BiasInitializer;
        m_BiasInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Convolution* Convolution::SetUseBias(bool useBias)
    {
        m_UseBias = useBias;
        return this;
    }
}

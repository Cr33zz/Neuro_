#include "Layers/Convolution.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Convolution::Convolution(LayerBase* inputLayer, int filterSize, int filtersNum, int stride, ActivationBase* activation, const string& name)
		: LayerBase(inputLayer, GetOutShape(inputLayer->OutputShape(), filterSize, filterSize, stride, filtersNum), activation, name.empty() ? GenerateName() : name)
	{
		FilterSize = filterSize;
		FiltersNum = filtersNum;
		Stride = stride;
	}

	//////////////////////////////////////////////////////////////////////////
	Convolution::Convolution(const Shape& inputShape, int filterSize, int filtersNum, int stride, ActivationBase* activation, const string& name)
		: LayerBase(inputShape, GetOutShape(inputShape, filterSize, filterSize, stride, filtersNum), activation, name.empty() ? GenerateName() : name)
	{
		FilterSize = filterSize;
		FiltersNum = filtersNum;
		Stride = stride;
	}

	//////////////////////////////////////////////////////////////////////////
	Convolution::Convolution()
	{

	}

	//////////////////////////////////////////////////////////////////////////
	Convolution::~Convolution()
	{
		delete KernelInitializer;
		delete BiasInitializer;
	}

	//////////////////////////////////////////////////////////////////////////
	void Convolution::OnInit()
	{
		__super::OnInit();

		Kernels = Tensor(Shape(FilterSize, FilterSize, InputShape().Depth(), FiltersNum));
		Bias = Tensor(Shape(m_OutputShape.Width(), m_OutputShape.Height(), FiltersNum));
		KernelsGradient = Tensor(Kernels.GetShape());
		BiasGradient = Tensor(Bias.GetShape());

		KernelInitializer->Init(Kernels, m_InputShapes[0].Length, m_OutputShape.Length);
		if (UseBias)
			BiasInitializer->Init(Bias, m_InputShapes[0].Length, m_OutputShape.Length);
	}

	//////////////////////////////////////////////////////////////////////////
	Neuro::LayerBase* Convolution::GetCloneInstance() const
	{
		return new Convolution();
	}

	//////////////////////////////////////////////////////////////////////////
	void Convolution::OnClone(const LayerBase& source)
	{
		__super::OnClone(source);

		auto& sourceConv = static_cast<const Convolution&>(source);
		Kernels = Tensor(sourceConv.Kernels);
		Bias = Tensor(sourceConv.Bias);
		UseBias = sourceConv.UseBias;
		FilterSize = sourceConv.FilterSize;
		FiltersNum = sourceConv.FiltersNum;
		Stride = sourceConv.Stride;
	}

	//////////////////////////////////////////////////////////////////////////
	void Convolution::FeedForwardInternal()
	{
		m_Inputs[0]->Conv2D(Kernels, Stride, Tensor::EPaddingType::Valid, m_Output);
		if (UseBias)
			m_Output.Add(Bias, m_Output);
	}

	//////////////////////////////////////////////////////////////////////////
	void Convolution::BackPropInternal(Tensor& outputGradient)
	{
		outputGradient.Conv2DInputsGradient(outputGradient, Kernels, Stride, Tensor::EPaddingType::Valid, m_InputsGradient[0]);
		outputGradient.Conv2DKernelsGradient(*m_Inputs[0], outputGradient, Stride, Tensor::EPaddingType::Valid, KernelsGradient);

		if (UseBias)
			BiasGradient.Add(outputGradient.SumBatches());
	}

	//////////////////////////////////////////////////////////////////////////
	Shape Convolution::GetOutShape(const Shape& inputShape, int filterWidth, int filterHeight, int stride, int filtersNum)
	{
		return Shape((int)floor((float)(inputShape.Width() - filterWidth) / stride + 1), (int)floor((float)(inputShape.Height() - filterHeight) / stride + 1), filtersNum);
	}

	//////////////////////////////////////////////////////////////////////////
	void Convolution::GetParametersAndGradients(vector<ParametersAndGradients>& result)
	{
		result.push_back(ParametersAndGradients(&Kernels, &KernelsGradient));

		if (UseBias)
			result.push_back(ParametersAndGradients(&Bias, &BiasGradient));
	}

	//////////////////////////////////////////////////////////////////////////
	void Convolution::CopyParametersTo(LayerBase& target, float tau) const
	{
		__super::CopyParametersTo(target, tau);

		auto& targetConv = static_cast<Convolution&>(target);
		Kernels.CopyTo(targetConv.Kernels, tau);
		Bias.CopyTo(targetConv.Bias, tau);
	}

	//////////////////////////////////////////////////////////////////////////
	int Convolution::GetParamsNum() const
	{
		return FilterSize * FilterSize * FiltersNum;
	}

	//////////////////////////////////////////////////////////////////////////
	const char* Convolution::ClassName() const
	{
		return "Conv";
	}

}

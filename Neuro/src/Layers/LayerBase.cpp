#include <sstream>

#include "Activations.h"
#include "Layers/LayerBase.h"
#include "Tensors/Shape.h"

namespace Neuro
{
	map<const char*, int> LayerBase::s_LayersCountPerType;

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase(LayerBase* inputLayer, const Shape& outputShape, ActivationBase* activation, const string& name)
		: LayerBase(outputShape, activation, name)
	{
		m_InputShapes.push_back(inputLayer->m_OutputShape);
		m_InputLayers.push_back(inputLayer);
		inputLayer->m_OutputLayers.push_back(this);
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase(const vector<LayerBase*>& inputLayers, const Shape& outputShape, ActivationBase* activation, const string& name)
		: LayerBase(outputShape, activation, name)
	{
		m_InputLayers.insert(m_InputLayers.end(), inputLayers.begin(), inputLayers.end());
		for (auto inLayer : inputLayers)
		{
			m_InputShapes.push_back(inLayer->m_OutputShape);
			inLayer->m_OutputLayers.push_back(this);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase(const Shape& inputShape, const Shape& outputShape, ActivationBase* activation, const string& name)
		: LayerBase(outputShape, activation, name)
	{
		m_InputShapes.push_back(inputShape);
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase(const vector<Shape>& inputShapes, const Shape& outputShape, ActivationBase* activation, const string& name)
		: LayerBase(outputShape, activation, name)
	{
		m_InputShapes = inputShapes;
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase(const Shape& outputShape, ActivationBase* activation, const string& name)
	{
		m_OutputShape = outputShape;
		m_Activation = activation;
		m_Name = name;
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase()
	{

	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase* LayerBase::Clone()
	{
		Init(); // make sure parameter matrices are created
		LayerBase* clone = GetCloneInstance();
		clone->OnClone(*this);
		return clone;
	}

	//////////////////////////////////////////////////////////////////////////
	void LayerBase::OnClone(const LayerBase& source)
	{
		m_InputShapes = source.m_InputShapes;
		m_OutputShape = source.m_OutputShape;
		m_Activation = source.m_Activation;
		m_Name = source.m_Name;
	}

	//////////////////////////////////////////////////////////////////////////
	void LayerBase::CopyParametersTo(LayerBase& target, float tau) const
	{
		assert(m_InputShapes == target.m_InputShapes && m_OutputShape == target.m_OutputShape && "Cannot copy parameters between incompatible layers.");
	}

	//////////////////////////////////////////////////////////////////////////
	void LayerBase::ExecuteFeedForward()
	{
		Shape outShape(m_OutputShape.Width(), m_OutputShape.Height(), m_OutputShape.Depth(), m_Inputs[0]->BatchSize());
		// shape comparison is required for cases when last batch has different size
		if (m_Output.GetShape() != outShape)
			m_Output = Tensor(outShape);

		//FeedForwardTimer.Start();
		FeedForwardInternal();
		//FeedForwardTimer.Stop();

		if (m_Activation)
		{
			//ActivationTimer.Start();
			m_Activation->Compute(m_Output, m_Output);
			//ActivationTimer.Stop();

			/*if (NeuralNetwork::DebugMode)
				Trace.WriteLine($"Activation({Activation.GetType().Name}) output:\n{Output}\n");*/
		}
	}

	//////////////////////////////////////////////////////////////////////////
	const Tensor* LayerBase::FeedForward(const Tensor* input)
	{
		if (!Initialized)
			Init();

		m_Inputs.resize(1);
		m_Inputs[0] = input;
		ExecuteFeedForward();
		return &m_Output;
	}

	//////////////////////////////////////////////////////////////////////////
	const Tensor* LayerBase::FeedForward(const vector<const Tensor*>& inputs)
	{
		if (!Initialized)
			Init();

		m_Inputs = inputs;
		ExecuteFeedForward();
		return &m_Output;
	}

	//////////////////////////////////////////////////////////////////////////
	vector<Tensor>& LayerBase::BackProp(Tensor& outputGradient)
	{
		m_InputsGradient.resize(m_InputShapes.size());

		for (int i = 0; i < (int)m_InputShapes.size(); ++i)
		{
			auto& inputShape = m_InputShapes[i];
			Shape deltaShape(inputShape.Width(), inputShape.Height(), inputShape.Depth(), outputGradient.BatchSize());
			if (m_InputsGradient[i].GetShape() != deltaShape)
				m_InputsGradient[i] = Tensor(deltaShape);
		}

		// apply derivative of our activation function to the errors computed by previous layer
		if (m_Activation)
		{
			//ActivationBackPropTimer.Start();
			m_Activation->Derivative(m_Output, outputGradient, outputGradient);
			//ActivationBackPropTimer.Stop();

			/*if (NeuralNetwork::DebugMode)
				Trace.WriteLine($"Activation({Activation.GetType().Name}) errors gradient:\n{outputGradient}\n");*/
		}

		//BackPropTimer.Start();
		BackPropInternal(outputGradient);
		//BackPropTimer.Stop();

		return m_InputsGradient;
	}

	//////////////////////////////////////////////////////////////////////////
	int LayerBase::GetParamsNum() const
	{
		return 0;
	}

	//////////////////////////////////////////////////////////////////////////
	void LayerBase::GetParametersAndGradients(vector<ParametersAndGradients>& result)
	{
	}

	//////////////////////////////////////////////////////////////////////////
	void LayerBase::Init()
	{
		if (Initialized)
			return;

		OnInit();
		Initialized = true;
	}

	//////////////////////////////////////////////////////////////////////////
	void LayerBase::OnInit()
	{
	}

	//////////////////////////////////////////////////////////////////////////
	string LayerBase::GenerateName() const
	{
		stringstream ss;
		ss << ClassName() << "_" << (++s_LayersCountPerType[ClassName()]);
		return ss.str();
	}
}

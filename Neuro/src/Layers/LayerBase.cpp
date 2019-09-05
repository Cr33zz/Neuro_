#include <sstream>

#include "Activations.h"
#include "Layers/LayerBase.h"
#include "Tensors/Shape.h"
#include "Tools.h"
#include "NeuralNetwork.h"

//#define LOG_OUTPUTS

namespace Neuro
{
	map<string, int> LayerBase::s_LayersCountPerType;

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase(const string& constructorName, LayerBase* inputLayer, const Shape& outputShape, ActivationBase* activation, const string& name)
		: LayerBase(constructorName, outputShape, activation, name)
	{
		m_InputShapes.push_back(inputLayer->m_OutputShape);
		m_InputLayers.push_back(inputLayer);
		inputLayer->m_OutputLayers.push_back(this);
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase(const string& constructorName, const vector<LayerBase*>& inputLayers, const Shape& outputShape, ActivationBase* activation, const string& name)
		: LayerBase(constructorName, outputShape, activation, name)
	{
		m_InputLayers.insert(m_InputLayers.end(), inputLayers.begin(), inputLayers.end());
		for (auto inLayer : inputLayers)
		{
			m_InputShapes.push_back(inLayer->m_OutputShape);
			inLayer->m_OutputLayers.push_back(this);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase(const string& constructorName, const Shape& inputShape, const Shape& outputShape, ActivationBase* activation, const string& name)
		: LayerBase(constructorName, outputShape, activation, name)
	{
		m_InputShapes.push_back(inputShape);
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase(const string& constructorName, const vector<Shape>& inputShapes, const Shape& outputShape, ActivationBase* activation, const string& name)
		: LayerBase(constructorName, outputShape, activation, name)
	{
		m_InputShapes = inputShapes;
	}

	//////////////////////////////////////////////////////////////////////////
    LayerBase::LayerBase(const string& constructorName, const Shape& outputShape, ActivationBase* activation, const string& name)
	{
		m_OutputShape = outputShape;
		m_Activation = activation;
        m_ClassName = ToLower(constructorName.substr(constructorName.find_last_of("::") + 1));
        m_Name = name.empty() ? GenerateName() : name;
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
	void LayerBase::ExecuteFeedForward(bool training)
	{
		Shape outShape(m_OutputShape.Width(), m_OutputShape.Height(), m_OutputShape.Depth(), m_Inputs[0]->Batch());
		// shape comparison is required for cases when last batch has different size
		if (m_Output.GetShape() != outShape)
			m_Output = Tensor(outShape, Name() + "/output");

#ifdef LOG_OUTPUTS
        for (int i = 0; i < (int)m_Inputs.size(); ++i)
            m_Inputs[i]->DebugDumpValues(Replace(Name() + "input_" + to_string(i) + "_step" + to_string(NeuralNetwork::g_DebugStep) + ".log", "/", "__"));
#endif

		m_FeedForwardTimer.Start();
		FeedForwardInternal(training);
		m_FeedForwardTimer.Stop();

#ifdef LOG_OUTPUTS
        m_Output.DebugDumpValues(Replace(m_Output.Name() + "_step" + to_string(NeuralNetwork::g_DebugStep) + ".log", "/", "__"));
#endif

		if (m_Activation)
		{
			m_ActivationTimer.Start();
			m_Activation->Compute(m_Output, m_Output);
			m_ActivationTimer.Stop();

#ifdef LOG_OUTPUTS
            m_Output.DebugDumpValues(Replace(Name() + "_activation_step" + to_string(NeuralNetwork::g_DebugStep) + ".log", "/", "__"));
#endif
		}
	}

	//////////////////////////////////////////////////////////////////////////
	const Tensor* LayerBase::FeedForward(const Tensor* input, bool training)
	{
		if (!Initialized)
			Init();

		m_Inputs.resize(1);
		m_Inputs[0] = input;
		ExecuteFeedForward(training);
		return &m_Output;
	}

	//////////////////////////////////////////////////////////////////////////
	const Tensor* LayerBase::FeedForward(const vector<const Tensor*>& inputs, bool training)
	{
		if (!Initialized)
			Init();

		m_Inputs = inputs;
		ExecuteFeedForward(training);
		return &m_Output;
	}

	//////////////////////////////////////////////////////////////////////////
	vector<Tensor>& LayerBase::BackProp(Tensor& outputGradient)
	{
		m_InputsGradient.resize(m_InputShapes.size());

		for (uint i = 0; i < (int)m_InputShapes.size(); ++i)
		{
			auto& inputShape = m_InputShapes[i];
			Shape deltaShape(inputShape.Width(), inputShape.Height(), inputShape.Depth(), outputGradient.Batch());
			if (m_InputsGradient[i].GetShape() != deltaShape)
				m_InputsGradient[i] = Tensor(deltaShape, Name() + "/input_" + to_string(i) + "_grad");
		}

		// apply derivative of our activation function to the errors computed by previous layer
		if (m_Activation)
		{
			m_ActivationBackPropTimer.Start();
			m_Activation->Derivative(m_Output, outputGradient, outputGradient);
			m_ActivationBackPropTimer.Stop();

#ifdef LOG_OUTPUTS
            outputGradient.DebugDumpValues(Replace(Name() + "_activation_grad_step" + to_string(NeuralNetwork::g_DebugStep) + ".log", "/", "__"));
#endif
		}

		m_BackPropTimer.Start();
		BackPropInternal(outputGradient);
		m_BackPropTimer.Stop();

#ifdef LOG_OUTPUTS
        for (int i = 0; i < (int)m_InputShapes.size(); ++i)
        {
            m_InputsGradient[i].DebugDumpValues(Replace(m_InputsGradient[i].Name() + "_step" + to_string(NeuralNetwork::g_DebugStep) + ".log", "/", "__"));
        }
#endif

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
		ss << m_ClassName << "_" << (++s_LayersCountPerType[m_ClassName]);
		return ss.str();
	}
}

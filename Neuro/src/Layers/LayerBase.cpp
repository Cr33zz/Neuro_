#include <sstream>

#include "Activations.h"
#include "Layers/LayerBase.h"
#include "Tensors/Shape.h"
#include "Tools.h"
#include "NeuralNetwork.h"
#include "Types.h"

namespace Neuro
{
	map<string, int> LayerBase::s_LayersCountPerType;

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase(const string& constructorName, LayerBase* inputLayer, const Shape& outputShape, ActivationBase* activation, const string& name)
		: LayerBase(constructorName, outputShape, activation, name)
	{
        Link(inputLayer);
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase(const string& constructorName, const vector<LayerBase*>& inputLayers, const Shape& outputShape, ActivationBase* activation, const string& name)
		: LayerBase(constructorName, outputShape, activation, name)
	{
        Link(inputLayers);
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
        m_Outputs.resize(1);
        m_OutputShapes.resize(1);
		m_OutputShapes[0] = outputShape;
		m_Activation = activation;
        m_ClassName = ToLower(constructorName.substr(constructorName.find_last_of("::") + 1));
        m_Name = name.empty() ? GenerateName() : name;
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
		m_OutputShapes = source.m_OutputShapes;
		m_Activation = source.m_Activation;
		m_Name = source.m_Name;
	}

    //////////////////////////////////////////////////////////////////////////
    void LayerBase::Link(LayerBase* inputLayer)
    {
        assert(inputLayer->m_OutputShapes.size() == 1);
        m_InputShapes.push_back(inputLayer->m_OutputShapes[0]);
        m_InputLayers.push_back(inputLayer);
        inputLayer->m_OutputLayers.push_back(this);
        OnLink();
    }

    //////////////////////////////////////////////////////////////////////////
    void LayerBase::Link(const vector<LayerBase*>& inputLayers)
    {
        m_InputLayers.insert(m_InputLayers.end(), inputLayers.begin(), inputLayers.end());
        for (auto inLayer : inputLayers)
        {
            assert(inLayer->m_OutputShapes.size() == 1);
            m_InputShapes.push_back(inLayer->m_OutputShapes[0]);
            inLayer->m_OutputLayers.push_back(this);
        }
        OnLink();
    }

    //////////////////////////////////////////////////////////////////////////
	void LayerBase::CopyParametersTo(LayerBase& target, float tau) const
	{
		assert(m_InputShapes == target.m_InputShapes && m_OutputShapes == target.m_OutputShapes && "Cannot copy parameters between incompatible layers.");
	}

	//////////////////////////////////////////////////////////////////////////
    void LayerBase::ExecuteFeedForward(bool training)
    {
        for (size_t o = 0; o < m_Outputs.size(); ++o)
        {
            Shape outShape(m_OutputShapes[o].Width(), m_OutputShapes[o].Height(), m_OutputShapes[o].Depth(), m_Inputs[0]->Batch());
            // shape comparison is required for cases when last batch has different size
            if (m_Outputs[o].GetShape() != outShape)
                m_Outputs[o] = Tensor(outShape, Name() + "/output_" + to_string(o));
        }

#ifdef LOG_OUTPUTS
        for (int i = 0; i < (int)m_Inputs.size(); ++i)
            m_Inputs[i]->DebugDumpValues(Replace(Name() + "_input_" + to_string(i) + "_step" + to_string(NeuralNetwork::g_DebugStep) + ".log", "/", "__"));
#endif

        m_FeedForwardTimer.Start();
        FeedForwardInternal(training);
        m_FeedForwardTimer.Stop();

#ifdef LOG_OUTPUTS
        for (size_t o = 0; o < m_Outputs.size(); ++o)
            m_Outputs[o].DebugDumpValues(Replace(m_Outputs[o].Name() + "_step" + to_string(NeuralNetwork::g_DebugStep) + ".log", "/", "__"));
#endif

        if (m_Activation)
        {
            for (size_t o = 0; o < m_Outputs.size(); ++o)
            {
                m_ActivationTimer.Start();
                m_Activation->Compute(m_Outputs[o], m_Outputs[o]);
                m_ActivationTimer.Stop();

#ifdef LOG_OUTPUTS
                m_Outputs[o].DebugDumpValues(Replace(m_Outputs[o].Name() + "_activation_step" + to_string(NeuralNetwork::g_DebugStep) + ".log", "/", "__"));
#endif
            }
        }
	}

	//////////////////////////////////////////////////////////////////////////
    const Tensor& LayerBase::FeedForward(const Tensor* input, bool training)
	{
		if (!Initialized)
			Init();

		m_Inputs.resize(1);
		m_Inputs[0] = input;
		ExecuteFeedForward(training);
		return m_Outputs[0];
	}

	//////////////////////////////////////////////////////////////////////////
	const Tensor& LayerBase::FeedForward(const tensor_ptr_vec_t& inputs, bool training)
	{
		if (!Initialized)
			Init();

		m_Inputs = inputs;
		ExecuteFeedForward(training);
		return m_Outputs[0];
	}

	//////////////////////////////////////////////////////////////////////////
    vector<Tensor>& LayerBase::BackProp(vector<Tensor>& outputGradients)
	{
		m_InputGradients.resize(m_InputShapes.size());

		for (uint32_t i = 0; i < (int)m_InputShapes.size(); ++i)
		{
			auto& inputShape = m_InputShapes[i];
			Shape deltaShape(inputShape.Width(), inputShape.Height(), inputShape.Depth(), outputGradients[0].Batch());
			if (m_InputGradients[i].GetShape() != deltaShape)
				m_InputGradients[i] = Tensor(deltaShape, Name() + "/input_" + to_string(i) + "_grad");
		}

		// apply derivative of our activation function to the errors computed by previous layer
		if (m_Activation)
		{
            for (size_t o = 0; o < outputGradients.size(); ++o)
            {
                m_ActivationBackPropTimer.Start();
                m_Activation->Derivative(m_Outputs[o], outputGradients[o], outputGradients[o]);
                m_ActivationBackPropTimer.Stop();

#ifdef LOG_OUTPUTS
                outputGradient.DebugDumpValues(Replace(Name() + "_activation_grad_step" + to_string(NeuralNetwork::g_DebugStep) + ".log", "/", "__"));
#endif
            }
		}

		m_BackPropTimer.Start();
		BackPropInternal(outputGradients);
		m_BackPropTimer.Stop();

#ifdef LOG_OUTPUTS
        for (int i = 0; i < (int)m_InputShapes.size(); ++i)
        {
            m_InputGradients[i].DebugDumpValues(Replace(m_InputGradients[i].Name() + "_step" + to_string(NeuralNetwork::g_DebugStep) + ".log", "/", "__"));
        }
#endif

		return m_InputGradients;
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
	string LayerBase::GenerateName() const
	{
        stringstream ss;
		ss << m_ClassName << "_" << (++s_LayersCountPerType[m_ClassName]);
		return ss.str();
	}
}

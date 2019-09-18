#include <sstream>

#include "Activations.h"
#include "Layers/LayerBase.h"
#include "Tensors/Shape.h"
#include "Tools.h"
#include "Models/ModelBase.h"
#include "Types.h"

namespace Neuro
{
	map<string, int> LayerBase::s_LayersCountPerType;

	//////////////////////////////////////////////////////////////////////////
    LayerBase::LayerBase(const string& constructorName, const string& name)
	{
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
		m_Name = source.m_Name;
	}

    //////////////////////////////////////////////////////////////////////////
    void LayerBase::OnLink(LayerBase* layer, bool input)
    {
        if (input)
        {
            InputShapes() = layer->OutputShapes();
            InputLayers().resize(1);
            InputLayers()[0] = layer;
        }
        else
        {
            OutputLayers().push_back(layer);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void LayerBase::OnLink(const vector<LayerBase*>& layers, bool input)
    {
        if (input)
        {
            auto& inputShapes = InputShapes();

            inputShapes.clear();
            for (auto inLayer : layers)
                inputShapes.push_back(inLayer->OutputShape());
            InputLayers() = layers;
        }
        else
            assert(false);
    }

    //////////////////////////////////////////////////////////////////////////
    void LayerBase::Link(LayerBase* inputLayer)
    {
        assert(!HasInputLayers());

        auto& inputShapes = InputShapes();
        auto& otherOutputShapes = inputLayer->OutputShapes();

        // check if we only need to latch outputs to inputs
        if (!inputShapes.empty() && inputShapes.size() == otherOutputShapes.size())
        {
            for (size_t i = 0; i < inputShapes.size(); ++i)
                assert(inputShapes[i] == otherOutputShapes[i]);
        }

        OnLink(inputLayer, true);
        inputLayer->OnLink(this, false);
    }

    //////////////////////////////////////////////////////////////////////////
    void LayerBase::Link(const vector<LayerBase*>& inputLayers)
    {
        assert(!HasInputLayers());

        InputLayers().insert(InputLayers().end(), inputLayers.begin(), inputLayers.end());
        InputShapes().clear();

        OnLink(inputLayers, true);

        for (auto inLayer : inputLayers)
        {
            assert(inLayer->OutputShapes().size() == 1);
            inLayer->OnLink(this, false);            
        }
    }

    //////////////////////////////////////////////////////////////////////////
	void LayerBase::CopyParametersTo(LayerBase& target, float tau) const
	{
		assert(InputShapes() == target.InputShapes() && OutputShapes() == target.OutputShapes() && "Cannot copy parameters between incompatible layers.");
	}

	//////////////////////////////////////////////////////////////////////////
    void LayerBase::ExecuteFeedForward(bool training)
    {
        auto& outputShapes = OutputShapes();
        auto& outputs = Outputs();
        auto& inputs = Inputs();

        for (size_t o = 0; o < outputs.size(); ++o)
        {
            Shape outShape(outputShapes[o].Width(), outputShapes[o].Height(), outputShapes[o].Depth(), inputs[0]->Batch());
            // shape comparison is required for cases when last batch has different size
            if (outputs[o].GetShape() != outShape)
                outputs[o] = Tensor(outShape, Name() + "/output_" + to_string(o));
        }

#       ifdef LOG_OUTPUTS
        for (auto i = 0; i < inputs.size(); ++i)
            inputs[i]->DebugDumpValues(Replace(Name() + "_input_" + to_string(i) + "_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
#       endif

        m_FeedForwardTimer.Start();
        FeedForwardInternal(training);
        m_FeedForwardTimer.Stop();

#       ifdef LOG_OUTPUTS
        for (auto o = 0; o < outputs.size(); ++o)
            outputs[o].DebugDumpValues(Replace(Name() + "_output_" + to_string(o) + "_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
#       endif

        if (Activation())
        {
            for (size_t o = 0; o < outputs.size(); ++o)
            {
                m_ActivationTimer.Start();
                Activation()->Compute(outputs[o], outputs[o]);
                m_ActivationTimer.Stop();

#               ifdef LOG_OUTPUTS
                outputs[o].DebugDumpValues(Replace(Name() + "_output_" + to_string(o) + "_activation_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
#               endif
            }
        }
	}

	//////////////////////////////////////////////////////////////////////////
    const Tensor& LayerBase::FeedForward(const Tensor* input, bool training)
	{
		if (!Initialized)
			Init();

        auto& inputs = Inputs();

        inputs.resize(1);
        inputs[0] = input;
		ExecuteFeedForward(training);
		return Outputs()[0];
	}

	//////////////////////////////////////////////////////////////////////////
	const Tensor& LayerBase::FeedForward(const tensor_ptr_vec_t& inputs, bool training)
	{
		if (!Initialized)
			Init();

        Inputs() = inputs;
		ExecuteFeedForward(training);
		return Outputs()[0];
	}

	//////////////////////////////////////////////////////////////////////////
    vector<Tensor>& LayerBase::BackProp(vector<Tensor>& outputsGradient)
	{
        auto& inputsGrad = InputsGradient();
        auto& inputShapes = InputShapes();
        auto& outputs = Outputs();

        inputsGrad.resize(inputShapes.size());

        if (!CanStopBackProp())
        {
            for (auto i = 0; i < inputShapes.size(); ++i)
            {
                auto& inputShape = inputShapes[i];
                Shape deltaShape(inputShape.Width(), inputShape.Height(), inputShape.Depth(), outputsGradient[0].Batch());
                if (inputsGrad[i].GetShape() != deltaShape)
                    inputsGrad[i] = Tensor(deltaShape, Name() + "/input_" + to_string(i) + "_grad");
            }

            // apply derivative of our activation function to the errors computed by previous layer
            if (Activation())
            {
                for (size_t o = 0; o < outputsGradient.size(); ++o)
                {
                    m_ActivationBackPropTimer.Start();
                    Activation()->Derivative(outputs[o], outputsGradient[o], outputsGradient[o]);
                    m_ActivationBackPropTimer.Stop();
#                   ifdef LOG_OUTPUTS
                    outputsGradient[o].DebugDumpValues(Replace(Name() + "_activation" + to_string(o) + "_grad_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
#                   endif
                }
            }

            m_BackPropTimer.Start();
            BackPropInternal(outputsGradient);
            m_BackPropTimer.Stop();
#           ifdef LOG_OUTPUTS
            for (auto i = 0; i < inputShapes.size(); ++i)
                inputsGrad[i].DebugDumpValues(Replace(inputsGrad[i].Name() + "_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
#           endif
        }

		return inputsGrad;
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
    vector<Tensor*> LayerBase::GetParams()
    {
        vector<Tensor*> params;

        vector<ParametersAndGradients> paramsAndGrads;
        GetParametersAndGradients(paramsAndGrads, false);
        for (auto i = 0; i < paramsAndGrads.size(); ++i)
            params.push_back(paramsAndGrads[i].Parameters);

        return params;
    }

    //////////////////////////////////////////////////////////////////////////
    bool LayerBase::CanStopBackProp() const
    {
        if (m_Trainable)
            return false;

        auto& inputLayers = InputLayers();

        if (inputLayers.empty())
            return true;

        for (auto inputLayer : inputLayers)
        {
            if (!inputLayer->CanStopBackProp())
                return false;
        }

        return true;
    }

    //////////////////////////////////////////////////////////////////////////
	string LayerBase::GenerateName() const
	{
        stringstream ss;
		ss << m_ClassName << "_" << (++s_LayersCountPerType[m_ClassName]);
		return ss.str();
	}
}

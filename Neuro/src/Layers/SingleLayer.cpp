#include <sstream>

#include "Layers/SingleLayer.h"
#include "Activations.h"
#include "Tensors/Shape.h"
#include "Tools.h"
#include "Models/ModelBase.h"
#include "Types.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    SingleLayer::SingleLayer(const string& constructorName, LayerBase* inputLayer, const Shape& outputShape, ActivationBase* activation, const string& name)
        : SingleLayer(constructorName, outputShape, activation, name)
    {
        LinkInput(inputLayer);
    }

    //////////////////////////////////////////////////////////////////////////
    SingleLayer::SingleLayer(const string& constructorName, const vector<LayerBase*>& inputLayers, const Shape& outputShape, ActivationBase* activation, const string& name)
        : SingleLayer(constructorName, outputShape, activation, name)
    {
        for (auto inputLayer : inputLayers)
            LinkInput(inputLayer);
    }

    //////////////////////////////////////////////////////////////////////////
    SingleLayer::SingleLayer(const string& constructorName, const Shape& inputShape, const Shape& outputShape, ActivationBase* activation, const string& name)
        : SingleLayer(constructorName, outputShape, activation, name)
    {
        m_InputsShapes.push_back(inputShape);
    }

    //////////////////////////////////////////////////////////////////////////
    SingleLayer::SingleLayer(const string& constructorName, const vector<Shape>& inputShapes, const Shape& outputShape, ActivationBase* activation, const string& name)
        : SingleLayer(constructorName, outputShape, activation, name)
    {
        m_InputsShapes = inputShapes;
    }

    //////////////////////////////////////////////////////////////////////////
    SingleLayer::SingleLayer(const string& constructorName, const Shape& outputShape, ActivationBase* activation, const string& name)
        : LayerBase(constructorName, name)
    {
        m_Outputs.resize(1);
        m_OutputsShapes.resize(1);
        m_OutputsShapes[0] = outputShape;
        m_Activation = activation;
    }

    //////////////////////////////////////////////////////////////////////////
    SingleLayer::~SingleLayer()
    {
        delete m_Outputs[0];
        for (auto inputGrad : m_InputsGradient)
            delete inputGrad;
    }

    //////////////////////////////////////////////////////////////////////////
    void SingleLayer::OnClone(const LayerBase& source)
    {
        __super::OnClone(source);
        
        auto& sourceSingleLayer = static_cast<const SingleLayer&>(source);
        m_InputsShapes = sourceSingleLayer.m_InputsShapes;
        m_OutputsShapes = sourceSingleLayer.m_OutputsShapes;
        m_Activation = sourceSingleLayer.m_Activation;
    }

    //////////////////////////////////////////////////////////////////////////
    void SingleLayer::OnLink(LayerBase* layer, bool input)
    {
        if (input)
        {
            m_InputsShapes.insert(m_InputsShapes.end(), layer->OutputShapes().begin(), layer->OutputShapes().end());
            m_InputLayers.push_back(layer);
        }
        else
        {
            m_OutputLayers.push_back(layer);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void SingleLayer::OnInit()
    {
        m_Outputs[0] = new Tensor(Name() + "/output_0");
        m_InputsGradient.resize(m_InputsShapes.size());

        for (auto i = 0; i < m_InputsShapes.size(); ++i)
        {
            auto& inputShape = m_InputsShapes[i];
            m_InputsGradient[i] = new Tensor(Name() + "/input_" + to_string(i) + "_grad");
        }
    }

    //////////////////////////////////////////////////////////////////////////
    int SingleLayer::InputOffset(const LayerBase* inputLayer) const
    {
        int offset = 0;
        for (auto layer : m_InputLayers)
        {
            if (inputLayer == layer)
                return offset;

            offset += (int)layer->Outputs().size();
        }
        return -offset;
    }

    //////////////////////////////////////////////////////////////////////////
    const tensor_ptr_vec_t& SingleLayer::FeedForward(const const_tensor_ptr_vec_t& inputs, bool training)
    {
        Init();

        m_Inputs = inputs;
        
        // resize is required for the very first batch and cases where last batch has different size
        m_Outputs[0]->Resize(Shape::From(m_OutputsShapes[0], m_Inputs[0]->Batch()));

#       ifdef LOG_OUTPUTS
        for (auto i = 0; i < m_Inputs.size(); ++i)
            m_Inputs[i]->DebugDumpValues(Replace(Name() + "_input_" + to_string(i) + "_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
#       endif

        m_FeedForwardTimer.Start();
        FeedForwardInternal(training);
        m_FeedForwardTimer.Stop();

#       ifdef LOG_OUTPUTS
        for (auto o = 0; o < m_Outputs.size(); ++o)
            m_Outputs[o].DebugDumpValues(Replace(Name() + "_output_" + to_string(o) + "_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
#       endif

        if (m_Activation)
        {
            for (size_t o = 0; o < m_Outputs.size(); ++o)
            {
                m_ActivationTimer.Start();
                m_Activation->Compute(*m_Outputs[o], *m_Outputs[o]);
                m_ActivationTimer.Stop();

#               ifdef LOG_OUTPUTS
                m_Outputs[o].DebugDumpValues(Replace(Name() + "_output_" + to_string(o) + "_activation_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
#               endif
            }
        }

        return m_Outputs;
    }

    //////////////////////////////////////////////////////////////////////////
    const tensor_ptr_vec_t& SingleLayer::BackProp(const tensor_ptr_vec_t& outputsGradient)
    {
        assert(outputsGradient.size() == 1);

        if (!CanStopBackProp())
        {
            for (auto i = 0; i < m_InputsShapes.size(); ++i)
                m_InputsGradient[i]->Resize(Shape::From(m_InputsShapes[i], outputsGradient[0]->Batch()));

            // apply derivative of our activation function to the errors computed by previous layer
            if (m_Activation)
            {
                m_ActivationBackPropTimer.Start();
                m_Activation->Derivative(*m_Outputs[0], *outputsGradient[0], *outputsGradient[0]);
                m_ActivationBackPropTimer.Stop();
#               ifdef LOG_OUTPUTS
                outputsGradient[0].DebugDumpValues(Replace(Name() + "_activation_0_grad_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
#               endif
            }

            m_BackPropTimer.Start();
            BackPropInternal(outputsGradient);
            m_BackPropTimer.Stop();
#           ifdef LOG_OUTPUTS
            for (auto i = 0; i < m_InputsShapes.size(); ++i)
                m_InputsGradient[i].DebugDumpValues(Replace(m_InputsGradient[i].Name() + "_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
#           endif
        }

        return m_InputsGradient;
    }

    //////////////////////////////////////////////////////////////////////////
    void SingleLayer::FeedForwardInternal(bool training)
    {
        assert(m_Inputs.size() == 1);
        // default implementation simply copy inputs over to outputs
        m_Inputs[0]->CopyTo(*m_Outputs[0]);
    }

    //////////////////////////////////////////////////////////////////////////
    void SingleLayer::BackPropInternal(const tensor_ptr_vec_t& outputsGradient)
    {
        assert(outputsGradient.size() == 1);
        // default implementation simply copy outputs gradient over to inputs gradient
        outputsGradient[0]->CopyTo(*m_InputsGradient[0]);
    }
}

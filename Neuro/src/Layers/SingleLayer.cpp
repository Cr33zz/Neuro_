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
        Link(inputLayer);
    }

    //////////////////////////////////////////////////////////////////////////
    SingleLayer::SingleLayer(const string& constructorName, const vector<LayerBase*>& inputLayers, const Shape& outputShape, ActivationBase* activation, const string& name)
        : SingleLayer(constructorName, outputShape, activation, name)
    {
        Link(inputLayers);
    }

    //////////////////////////////////////////////////////////////////////////
    SingleLayer::SingleLayer(const string& constructorName, const Shape& inputShape, const Shape& outputShape, ActivationBase* activation, const string& name)
        : SingleLayer(constructorName, outputShape, activation, name)
    {
        m_InputShape = inputShape;
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
        m_InputShape = sourceSingleLayer.m_InputShape;
        m_OutputsShapes = sourceSingleLayer.m_OutputsShapes;
        m_Activation = sourceSingleLayer.m_Activation;
    }

    //////////////////////////////////////////////////////////////////////////
    void SingleLayer::OnLinkInput(const vector<LayerBase*>& inputLayers)
    {
        // all output shapes must match
        Shape firstShape = inputLayers[0]->OutputShape();
        
        assert(!m_InputShape.IsValid() || m_InputShape == firstShape);

        for (size_t i = 1; i < inputLayers.size(); ++i)
            assert(firstShape == inputLayers[i]->OutputShape());

        m_InputShape = firstShape;
        m_InputLayers.insert(m_InputLayers.end(), inputLayers.begin(), inputLayers.end());
    }

    //////////////////////////////////////////////////////////////////////////
    void SingleLayer::OnLinkOutput(LayerBase* outputLayer)
    {
        m_OutputLayers.push_back(outputLayer);
    }

    //////////////////////////////////////////////////////////////////////////
    void SingleLayer::OnInit(bool initValues)
    {
        m_Outputs.resize(m_OutputsShapes.size());
        for (size_t i = 0; i < m_Outputs.size(); ++i)
            m_Outputs[i] = new Tensor(m_OutputsShapes[i], Name() + "/output_0");
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

        if (m_InputsGradient.size() != inputs.size())
        {
            m_InputsGradient.resize(inputs.size());
            for (auto i = 0; i < inputs.size(); ++i)
                m_InputsGradient[i] = new Tensor(Name() + "/input_" + to_string(i) + "_grad");
        }
        for (size_t i = 0; i < inputs.size(); ++i)
            m_InputsGradient[i]->Resize(inputs[i]->GetShape());
        
        // resize is required for the very first batch and cases where last batch has different size
        for (size_t i = 0; i < m_OutputsShapes.size(); ++i)
            m_Outputs[i]->Resize(Shape::From(m_OutputsShapes[i], m_Inputs[0]->Batch()));

#       ifdef LOG_OUTPUTS
        for (auto i = 0; i < m_Inputs.size(); ++i)
            m_Inputs[i]->DebugDumpValues(Replace(Name() + "_input_" + to_string(i) + "_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
#       endif

        m_FeedForwardTimer.Start();
        FeedForwardInternal(training);
        m_FeedForwardTimer.Stop();

#       ifdef LOG_OUTPUTS
        for (size_t i = 0; i < m_Outputs.size(); ++i)
            m_Outputs[i]->DebugDumpValues(Replace(Name() + "_output_" + to_string(i) + "_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
#       endif

        if (m_Activation)
        {
            for (size_t i = 0; i < m_Outputs.size(); ++i)
            {
                m_ActivationTimer.Start();
                m_Activation->Compute(*m_Outputs[i], *m_Outputs[i]);
                m_ActivationTimer.Stop();

#               ifdef LOG_OUTPUTS
                m_Outputs[i]->DebugDumpValues(Replace(Name() + "_output_" + to_string(i) + "_activation_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
#               endif
            }
        }

        return m_Outputs;
    }

    //////////////////////////////////////////////////////////////////////////
    const tensor_ptr_vec_t& SingleLayer::BackProp(const tensor_ptr_vec_t& outputsGradient)
    {
        if (!CanStopBackProp())
        {
            // apply derivative of our activation function to the errors computed by previous layer
            if (m_Activation)
            {
                for (size_t i = 0; i < m_Outputs.size(); ++i)
                {
                    m_ActivationBackPropTimer.Start();
                    m_Activation->Derivative(*m_Outputs[i], *outputsGradient[i], *outputsGradient[i]);
                    m_ActivationBackPropTimer.Stop();
#                   ifdef LOG_OUTPUTS
                    outputsGradient[i]->DebugDumpValues(Replace(Name() + "_activation_" + to_string(i) + "_grad_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
#                   endif
                }
            }

            m_BackPropTimer.Start();
            BackPropInternal(outputsGradient);
            m_BackPropTimer.Stop();

#           ifdef LOG_OUTPUTS
            for (size_t i = 0; i < m_InputsGradient.size(); ++i)
                m_InputsGradient[i]->DebugDumpValues(Replace(Name() + "_input_" + to_string(i) + "_grad_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
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

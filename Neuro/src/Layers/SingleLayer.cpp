#include <sstream>

#include "Types.h"
#include "Layers/SingleLayer.h"
#include "Activations.h"
#include "Tensors/Shape.h"
#include "Tools.h"
#include "Models/ModelBase.h"
#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/NameScope.h"

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
        if (m_InputLayers.empty())
        {
            NameScope layerScope(Name());
            m_InputOps.push_back(new Placeholder(m_InputShape, "input"));
        }
        else
        {
            for (auto inLayer : m_InputLayers)
                m_InputOps.insert(m_InputOps.end(), inLayer->OutputOps().begin(), inLayer->OutputOps().end());
        }

        m_Outputs.resize(m_OutputsShapes.size());
        m_OutputOps.resize(m_Outputs.size());

        InitOps(initValues);
        
        if (m_Activation)
        {
            for (size_t i = 0; i < m_Outputs.size(); ++i)
                m_OutputOps[i] = m_Activation->Build(m_OutputOps[i]);
        }
    }
}

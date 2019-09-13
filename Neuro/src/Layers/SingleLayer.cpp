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
        m_InputShapes.push_back(inputShape);
    }

    //////////////////////////////////////////////////////////////////////////
    SingleLayer::SingleLayer(const string& constructorName, const vector<Shape>& inputShapes, const Shape& outputShape, ActivationBase* activation, const string& name)
        : SingleLayer(constructorName, outputShape, activation, name)
    {
        m_InputShapes = inputShapes;
    }

    //////////////////////////////////////////////////////////////////////////
    SingleLayer::SingleLayer(const string& constructorName, const Shape& outputShape, ActivationBase* activation, const string& name)
        : LayerBase(constructorName, name)
    {
        m_Outputs.resize(1);
        m_OutputShapes.resize(1);
        m_OutputShapes[0] = outputShape;
        m_Activation = activation;
    }

    //////////////////////////////////////////////////////////////////////////
    void SingleLayer::OnClone(const LayerBase& source)
    {
        __super::OnClone(source);
        
        auto& sourceSingleLayer = static_cast<const SingleLayer&>(source);
        m_InputShapes = sourceSingleLayer.m_InputShapes;
        m_OutputShapes = sourceSingleLayer.m_OutputShapes;
        m_Activation = sourceSingleLayer.m_Activation;
    }
}

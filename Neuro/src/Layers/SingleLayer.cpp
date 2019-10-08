#include <sstream>

#include "Layers/SingleLayer.h"
#include "Activations.h"
#include "Tensors/Shape.h"
#include "Tools.h"
#include "Models/ModelBase.h"
#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/NameScope.h"
#include "ComputationalGraph/Operations/DumpOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    SingleLayer::SingleLayer(const string& constructorName, const Shape& inputShape, ActivationBase* activation, const string& name)
        : LayerBase(constructorName, inputShape, name), m_Activation(activation)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    SingleLayer::SingleLayer(const string& constructorName, ActivationBase* activation, const string& name)
        : SingleLayer(constructorName, Shape(), activation, name)
    {
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
        /*m_InputShape = sourceSingleLayer.m_InputShape;
        m_OutputsShapes = sourceSingleLayer.m_OutputsShapes;
        m_Activation = sourceSingleLayer.m_Activation;*/
    }
}

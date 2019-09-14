#include "Layers/Reshape.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Reshape::Reshape(LayerBase* inputLayer, const Shape& shape, const string& name)
        : Reshape(__FUNCTION__, inputLayer, shape, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Reshape::Reshape(const Shape& inputShape, const Shape& shape, const string& name)
        : Reshape(__FUNCTION__, inputShape, shape, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Reshape::Reshape(const Shape& shape, const string& name)
        : Reshape(__FUNCTION__, shape, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Reshape::Reshape(const string& constructorName, LayerBase* inputLayer, const Shape& shape, const string& name)
        : SingleLayer(constructorName, inputLayer, shape, nullptr, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Reshape::Reshape(const string& constructorName, const Shape& inputShape, const Shape& shape, const string& name)
        : SingleLayer(constructorName, inputShape, shape, nullptr, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Reshape::Reshape(const string& constructorName, const Shape& shape, const string& name)
        : SingleLayer(constructorName, shape, nullptr, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Reshape::GetCloneInstance() const
    {
        return new Reshape();
    }
}

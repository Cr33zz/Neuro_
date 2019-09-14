#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class Reshape : public SingleLayer
    {
    public:
        Reshape(LayerBase* inputLayer, const Shape& shape, const string& name = "");
        // This constructor should only be used for input layer
        Reshape(const Shape& inputShape, const Shape& shape, const string& name = "");
        Reshape(const Shape& shape, const string& name = "");

    protected:
        Reshape(const string& constructorName, LayerBase* inputLayer, const Shape& shape, const string& name);
        Reshape(const string& constructorName, const Shape& inputShape, const Shape& shape, const string& name);
        Reshape(const string& constructorName, const Shape& shape, const string& name);
        Reshape() {}

        virtual LayerBase* GetCloneInstance() const override;
    };
}

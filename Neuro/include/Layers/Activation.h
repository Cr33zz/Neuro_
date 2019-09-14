#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class Activation : public SingleLayer
    {
    public:
        Activation(LayerBase* inputLayer, ActivationBase* activation, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        Activation(ActivationBase* activation, const string& name = "");
        // This constructor should only be used for input layer
        Activation(const Shape& inputShape, ActivationBase* activation, const string& name = "");

    protected:
        Activation();

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnLink(LayerBase* layer, bool input) override;
    };
}

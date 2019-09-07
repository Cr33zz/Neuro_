#pragma once

#include "Layers/LayerBase.h"

namespace Neuro
{
    class Activation : public LayerBase
    {
    public:
        Activation(LayerBase* inputLayer, ActivationBase* activation, const string& name = "");
        // This constructor should only be used for input layer
        Activation(const Shape& inputShape, ActivationBase* activation, const string& name = "");

    protected:
        Activation();

        virtual LayerBase* GetCloneInstance() const override;
    };
}

#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class Activation : public SingleLayer
    {
    public:
        Activation(ActivationBase* activation, const string& name = "");
        // This constructor should only be used for input layer
        Activation(const Shape& inputShape, ActivationBase* activation, const string& name = "");

    protected:
        Activation();

        virtual LayerBase* GetCloneInstance() const override;        
        
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputs, TensorLike* training) override;
    };
}

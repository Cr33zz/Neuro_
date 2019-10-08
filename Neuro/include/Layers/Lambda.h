#pragma once

#include "Layers/LayerBase.h"

namespace Neuro
{
    // This layer should only be used when we want to combine raw input with output of another layer
    // somewhere inside a network
    /*class Lambda : public SingleLayer
    {
    public:
        Lambda(const Shape& inputShape, const string& name = "");

    protected:
        Lambda();

        virtual LayerBase* GetCloneInstance() const override;

        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputNodes, TensorLike* training) override;
    };*/
}

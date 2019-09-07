#pragma once

#include "Layers/LayerBase.h"

namespace Neuro
{
    class Reshape : public LayerBase
    {
    public:
        Reshape(LayerBase* inputLayer, const Shape& shape, const string& name = "");
        // This constructor should only be used for input layer
        Reshape(const Shape& inputShape, const Shape& shape, const string& name = "");

    protected:
        Reshape(const string& constructorName, LayerBase* inputLayer, const Shape& shape, const string& name);
        Reshape(const string& constructorName, const Shape& inputShape, const Shape& shape, const string& name);
        Reshape(const string& constructorName, const Shape& shape, const string& name);
        Reshape() {}

        virtual LayerBase* GetCloneInstance() const override;
        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(vector<Tensor>& outputGradients) override;
    };
}

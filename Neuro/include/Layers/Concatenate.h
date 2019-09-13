#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class Concatenate : public SingleLayer
    {
    public:
        Concatenate(const vector<LayerBase*>& inputLayers, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        Concatenate(const string& name = "");

    protected:
        Concatenate(bool) {}

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnLink(const vector<LayerBase*>& layers, bool input) override;
        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(vector<Tensor>& outputsGradient) override;
    };
}

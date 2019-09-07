#pragma once

#include "Layers/LayerBase.h"

namespace Neuro
{
    class Concatenate : public LayerBase
    {
    public:
        Concatenate(const vector<LayerBase*>& inputLayers, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        Concatenate(const string& name = "");

    protected:
        Concatenate(bool) {}

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnLink() override;
        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(vector<Tensor>& outputGradients) override;
    };
}

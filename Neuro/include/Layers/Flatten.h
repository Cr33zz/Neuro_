#pragma once

#include "Layers/Reshape.h"

namespace Neuro
{
    class Flatten : public Reshape
    {
	public:
        Flatten(LayerBase* inputLayer, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        Flatten(const string& name = "");
        // This constructor should only be used for input layer
        Flatten(const Shape& inputShape, const string& name = "");

	protected:
        Flatten(bool) {}

		virtual LayerBase* GetCloneInstance() const override;
        virtual void OnLinkInput(const vector<LayerBase*>& inputLayers) override;
    };
}

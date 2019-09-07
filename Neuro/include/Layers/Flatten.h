#pragma once

#include "Layers/Reshape.h"

namespace Neuro
{
    class Flatten : public Reshape
    {
	public:
        Flatten(LayerBase* inputLayer, const string& name = "");
        // This constructor should only be used for input layer
        Flatten(const Shape& inputShape, const string& name = "");

	protected:
        Flatten();

		virtual LayerBase* GetCloneInstance() const override;
	};
}

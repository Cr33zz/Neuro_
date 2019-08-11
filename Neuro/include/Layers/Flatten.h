#pragma once

#include "Layers/LayerBase.h"

namespace Neuro
{
    class Flatten : public LayerBase
    {
	public:
        Flatten(LayerBase* inputLayer, const string& name = "");
        // This constructor should only be used for input layer
        Flatten(const Shape& inputShape, const string& name = "");

		const char* ClassName() const;

	protected:
        Flatten();

		virtual LayerBase* GetCloneInstance() const override;
		virtual void FeedForwardInternal() override;
		virtual void BackPropInternal(Tensor& outputGradient) override;
	};
}

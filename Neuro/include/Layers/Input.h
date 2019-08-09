﻿#pragma once

#include "Layers/LayerBase.h"

namespace Neuro
{
    // This layer should only be used when we want to combine raw input with output of another layer
    // somewhere inside a network
    class Input : public LayerBase
    {
	public:
        Input(const Shape& inputShape, const string& name = "");

		virtual const char* ClassName() const override;

	protected:
        Input();

		virtual LayerBase* GetCloneInstance() const override;
		virtual void FeedForwardInternal() override;
		virtual void BackPropInternal(Tensor& outputGradient) override;
	};
}

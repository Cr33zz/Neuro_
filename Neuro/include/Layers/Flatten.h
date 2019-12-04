#pragma once

#include "Layers/Reshape.h"

namespace Neuro
{
    class Flatten : public Reshape
    {
	public:
        Flatten(const string& name = "");
        // This constructor should only be used for input layer
        Flatten(const Shape& inputShape, const string& name = "");

	protected:
		virtual LayerBase* GetCloneInstance() const override;

        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputs) override;
    };
}

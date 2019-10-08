#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class Placeholder;

    // This layer should only be used when we want to combine raw input with output of another layer
    // somewhere inside a network
    class Input : public LayerBase
    {
	public:
        Input(const Shape& inputShape, const string& name = "");

	protected:
        Input();

		virtual LayerBase* GetCloneInstance() const override;

        virtual void Build(const vector<Shape>& inputShapes) override;
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputNodes, TensorLike* training) override;

    private:
        Placeholder* m_Placeholder = nullptr;
	};
}

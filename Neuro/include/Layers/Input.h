#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class TensorLike;

    // This layer should only be used when we want to combine raw input with output of another layer
    // somewhere inside a network
    class Input : public LayerBase
    {
	public:
        Input(const Shape& inputShape, const string& name = "");
        Input(TensorLike* input, const string& name = "");

	protected:
        Input();

		virtual LayerBase* GetCloneInstance() const override;

        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputNodes, TensorLike* training) override;

    private:
        void InitInput(TensorLike* input);

        TensorLike* m_Input = nullptr;
	};
}

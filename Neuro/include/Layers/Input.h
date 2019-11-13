#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class TensorLike;

    // Models require input layers to be first ones. Sequential model will automatically create input when adding first non-input layer to it.
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
